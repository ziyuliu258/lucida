#!/usr/bin/env python3
"""Evaluate saved checkpoints on all currently safe selected GPUs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from lucida_mini.evaluation import rollout_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def current_closed_protocol(directory: Path) -> bool:
    path = directory / "closed_loop.json"
    if not path.is_file():
        return False
    return json.loads(path.read_text()).get("protocol") == rollout_protocol()


def gpu_free_memory() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values: dict[int, int] = {}
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        values[int(row[0])] = int(row[1])
    return values


def run_kind(
    kind: str,
    step: int,
    devices: list[int],
    args: argparse.Namespace,
) -> None:
    root = args.run_dir / f"{kind}-{step}"
    root.mkdir(parents=True, exist_ok=True)
    merged_rows = root / ("per_turn.csv" if kind == "teacher-forced" else "per_trajectory.csv")
    summary = root / "summary.csv"
    expected_rows = 207 if kind == "teacher-forced" else 50
    if merged_rows.is_file() and summary.is_file():
        with merged_rows.open(newline="") as handle:
            existing_count = sum(1 for _ in csv.DictReader(handle))
        with summary.open(newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
        if (
            existing_count == expected_rows and len(summary_rows) == 1
            and (kind == "teacher-forced" or current_closed_protocol(root))
        ):
            print(f"step={step} {kind} already complete; reusing {existing_count} fixed cases", flush=True)
            return

    script = PROJECT_ROOT / (
        "scripts/evaluate_teacher_forced.py"
        if kind == "teacher-forced"
        else "scripts/evaluate_closed_loop.py"
    )
    worker_outputs = [root / f"shard_{index:02d}" for index in range(len(devices))]
    processes: list[tuple[subprocess.Popen, object, Path]] = []
    trajectories = json.loads(args.manifest.read_text())["trajectories"]
    for index, (physical_device, output_dir) in enumerate(zip(devices, worker_outputs)):
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_data = output_dir / ("per_turn.csv" if kind == "teacher-forced" else "per_trajectory.csv")
        expected: dict[str, int] = {}
        for trajectory_index, trajectory in enumerate(trajectories):
            if trajectory_index % len(devices) != index:
                continue
            if kind == "teacher-forced":
                count = sum(bool(turn["supervise"]) for turn in trajectory["turns"])
                if count:
                    expected[trajectory["trajectory_id"]] = count
            else:
                expected[trajectory["trajectory_id"]] = 1
        actual: dict[str, int] = {}
        if shard_data.is_file() and (output_dir / "summary.csv").is_file():
            with shard_data.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    trajectory_id = row["trajectory_id"]
                    actual[trajectory_id] = actual.get(trajectory_id, 0) + 1
        if actual == expected and (kind == "teacher-forced" or current_closed_protocol(output_dir)):
            print(f"step={step} {kind} shard {index} already complete; reusing", flush=True)
            continue
        stale_rollouts = output_dir / "rollouts"
        if kind == "eval" and stale_rollouts.exists():
            destination = output_dir / "unused_rollouts"
            suffix = 1
            while destination.exists():
                destination = output_dir / f"unused_rollouts_{suffix}"
                suffix += 1
            stale_rollouts.rename(destination)
        command = [
            sys.executable,
            str(script),
            "--manifest",
            str(args.manifest),
            "--dataset-root",
            str(args.dataset_root),
            "--base-model",
            str(args.base_model),
            "--output-dir",
            str(output_dir),
            "--step",
            str(step),
            "--shard-count",
            str(len(devices)),
            "--shard-index",
            str(index),
            "--vision-max-pixels",
            str(args.vision_max_pixels),
            "--vision-min-pixels",
            str(args.vision_min_pixels),
        ]
        if not args.base_only:
            command.extend(["--adapter", str(args.run_dir / f"checkpoint-{step}")])
        if kind == "teacher-forced":
            command.extend(
                [
                    "--device",
                    "cuda:0",
                    "--max-sequence-length",
                    str(args.max_sequence_length),
                ]
            )
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(physical_device),
                # EGL's first eight devices match nvidia-smi's physical GPU
                # indices. Later DRM-device entries require /dev/dri
                # permissions this account lacks.
                "EGL_DEVICE_ID": str(physical_device),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "2",
            }
        )
        log = (output_dir / "worker.log").open("w")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log, output_dir))

    failures = []
    for process, log, output_dir in processes:
        code = process.wait()
        log.close()
        if code != 0:
            failures.append((output_dir, code))
    if failures:
        raise RuntimeError(f"{kind} shard failures: {failures}; see worker.log files")

    # Keep old outputs for inspection but exclude out-of-range shard IDs when
    # the available device count changed since a prior attempt.
    for directory in root.glob("shard_[0-9][0-9]"):
        if int(directory.name.split("_")[-1]) >= len(devices):
            destination = root / f"unused_{kind}_{directory.name}"
            suffix = 1
            while destination.exists():
                destination = root / f"unused_{kind}_{directory.name}_{suffix}"
                suffix += 1
            directory.rename(destination)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/merge_eval_shards.py"),
            "teacher" if kind == "teacher-forced" else "closed",
            "--shards-dir",
            str(root),
            "--output-dir",
            str(root),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print(f"step={step} {kind} complete across {len(devices)} GPUs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument(
        "--base-only", action="store_true",
        help="Evaluate the unmodified pretrained base at step 0 (no LoRA adapter).",
    )
    parser.add_argument("--devices", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--minimum-free-mib", type=int, default=9000)
    parser.add_argument("--vision-max-pixels", type=int, default=393216)
    parser.add_argument("--vision-min-pixels", type=int, default=65536)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.manifest = args.manifest.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.base_model = args.base_model.resolve()
    candidates = [int(value) for value in args.devices.split(",") if value.strip()]

    steps = [0] if args.base_only else args.steps
    for step in steps:
        checkpoint = args.run_dir / f"checkpoint-{step}"
        if not args.base_only and not checkpoint.is_dir():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
        free = gpu_free_memory()
        devices = [
            device
            for device in candidates
            if free.get(device, 0) >= args.minimum_free_mib
        ]
        if not devices:
            raise RuntimeError(
                f"no selected GPU has {args.minimum_free_mib} MiB free; snapshot={free}"
            )
        print(
            f"step={step} evaluation devices={devices}; free MiB="
            + ",".join(f"{device}:{free[device]}" for device in devices),
            flush=True,
        )
        run_kind("teacher-forced", step, devices, args)
        run_kind("eval", step, devices, args)

    teacher_rows = []
    completed_teacher_summaries = sorted(
        args.run_dir.glob("teacher-forced-*/summary.csv"),
        key=lambda path: int(path.parent.name.removeprefix("teacher-forced-")),
    )
    for summary_path in completed_teacher_summaries:
        step = int(summary_path.parent.name.removeprefix("teacher-forced-"))
        with summary_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader)
        teacher_rows.append({
            "step": int(step),
            "full_train_loss": float(row["teacher_forced_mean_ce"]),
            "canonical_exact_match_rate": float(row["canonical_exact_match_rate"]),
            "action_type_match_rate": float(row["action_type_match_rate"]),
            "valid_gizmoact_xml_rate": float(row["valid_gizmoact_xml_rate"]),
        })
    with (args.run_dir / "teacher_forced_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(teacher_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(teacher_rows)
    print(f"wrote teacher-forced checkpoint metrics to {args.run_dir / 'teacher_forced_metrics.csv'}")


if __name__ == "__main__":
    main()
