#!/usr/bin/env python3
"""Merge independently evaluated fixed-set shards into canonical reports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from lucida_mini.evaluation import full_loop_success


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_teacher(root: Path, output: Path) -> None:
    shard_dirs = sorted(root.glob("shard_*"))
    if not shard_dirs:
        raise FileNotFoundError(f"no teacher-forced shards under {root}")
    rows: list[dict[str, str]] = []
    type_counts: dict[str, int] = {}
    config = None
    for directory in shard_dirs:
        fields, part = read_csv(directory / "per_turn.csv")
        if part:
            rows.extend(part)
        payload = json.loads((directory / "summary.json").read_text())
        config = config or payload.get("config")
        for name, count in payload.get("gold_action_type_counts", {}).items():
            type_counts[name] = type_counts.get(name, 0) + int(count)
    if not rows:
        raise ValueError("teacher-forced shards contained no turns")
    rows.sort(key=lambda row: (row["trajectory_id"], int(row["turn_index"])))
    if len(rows) != 207:
        raise ValueError(f"expected 207 supervised turns, found {len(rows)}")
    total_nll = sum(float(row["teacher_forced_nll_sum"]) for row in rows)
    total_tokens = sum(int(row["teacher_forced_token_count"]) for row in rows)
    count = len(rows)
    summary = {
        "step": int(rows[0]["step"]),
        "supervised_turns": count,
        "teacher_forced_token_count": total_tokens,
        "teacher_forced_mean_ce": total_nll / total_tokens,
        "mean_per_turn_ce": sum(float(row["teacher_forced_token_ce"]) for row in rows) / count,
        "valid_gizmoact_xml_rate": sum(float(row["valid_gizmoact_xml"]) for row in rows) / count,
        "action_type_match_rate": sum(float(row["action_type_match"]) for row in rows) / count,
        "canonical_exact_match_rate": sum(float(row["canonical_exact_match"]) for row in rows) / count,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_turn.csv", list(rows[0]), rows)
    write_csv(output / "summary.csv", list(summary), [summary])
    context_rows = []
    for context_id in sorted({row["context_id"] for row in rows}):
        subset = [row for row in rows if row["context_id"] == context_id]
        tokens = sum(int(row["teacher_forced_token_count"]) for row in subset)
        nll = sum(float(row["teacher_forced_nll_sum"]) for row in subset)
        context_rows.append({
            "context_id": context_id,
            "supervised_turns": len(subset),
            "teacher_forced_token_count": tokens,
            "teacher_forced_mean_ce": nll / tokens,
            "valid_gizmoact_xml_rate": sum(float(row["valid_gizmoact_xml"]) for row in subset) / len(subset),
            "action_type_match_rate": sum(float(row["action_type_match"]) for row in subset) / len(subset),
            "canonical_exact_match_rate": sum(float(row["canonical_exact_match"]) for row in subset) / len(subset),
        })
    write_csv(output / "per_context.csv", list(context_rows[0]), context_rows)
    (output / "summary.json").write_text(
        json.dumps({"summary": summary, "gold_action_type_counts": type_counts, "config": config}, indent=2) + "\n"
    )


def merge_closed(root: Path, output: Path) -> None:
    shard_dirs = sorted(root.glob("shard_*"))
    if not shard_dirs:
        raise FileNotFoundError(f"no closed-loop shards under {root}")
    rows: list[dict[str, str]] = []
    rollouts: list[dict] = []
    protocols: list[dict | None] = []
    for directory in shard_dirs:
        _fields, part = read_csv(directory / "per_trajectory.csv")
        rows.extend(part)
        payload = json.loads((directory / "closed_loop.json").read_text())
        protocols.append(payload.get("protocol"))
        rollouts.extend(payload["rollouts"])
    if any(protocol != protocols[0] for protocol in protocols):
        raise ValueError("cannot merge closed-loop shards from different evaluation protocols")
    rows.sort(key=lambda row: row["trajectory_id"])
    rollouts.sort(key=lambda row: row["trajectory_id"])
    if len(rows) != 50 or len(rollouts) != 50:
        raise ValueError(f"expected 50 closed-loop cases, got {len(rows)} rows and {len(rollouts)} rollouts")
    # The plan defines full closed-loop success as valid actions plus an
    # explicit stop, with both translational/scale alignment and rotation
    # within tolerance. Recompute this aggregate from the primitive per-case
    # measurements instead of trusting a worker's derived convenience field.
    success_by_id = {}
    for row in rows:
        success = full_loop_success(row)
        row["closed_loop_success"] = str(success)
        success_by_id[row["trajectory_id"]] = success
    for rollout in rollouts:
        rollout["metrics"]["closed_loop_success"] = success_by_id[rollout["trajectory_id"]]
    fields = list(rows[0])
    numeric_fields = [field for field in fields if field not in {"trajectory_id", "context_id"}]
    summary = {field: float(np.mean([float(row[field]) for row in rows])) for field in numeric_fields}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_trajectory.csv", fields, rows)
    write_csv(output / "summary.csv", list(summary), [summary])
    context_rows = []
    for context_id in sorted({row["context_id"] for row in rows}):
        subset = [row for row in rows if row["context_id"] == context_id]
        context_rows.append({
            "context_id": context_id,
            "trajectories": len(subset),
            **{
                field: float(np.mean([float(row[field]) for row in subset]))
                for field in numeric_fields
                if field != "step"
            },
        })
    write_csv(output / "per_context.csv", list(context_rows[0]), context_rows)
    (output / "closed_loop.json").write_text(
        json.dumps({"protocol": protocols[0], "summary": summary, "rollouts": rollouts}, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("teacher", "closed"))
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "teacher":
        merge_teacher(args.shards_dir, args.output_dir)
    else:
        merge_closed(args.shards_dir, args.output_dir)
    print(f"merged {args.kind} evaluation shards from {args.shards_dir} into {args.output_dir}")


if __name__ == "__main__":
    main()
