#!/usr/bin/env python3
"""Recompute 3D IoU from saved closed-loop actions without model inference.

Closed-loop evaluations persist every parsed model action in ``closed_loop.json``.
This utility replays those actions from the frozen manifest and replaces only
their pose IoU values.  It is useful when improving the deterministic geometry
metric while preserving the exact rollout that was already evaluated.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from lucida_mini.actions import parse_action
from lucida_mini.expert import execute_action
from lucida_mini.metrics import oriented_box_iou
from lucida_mini.pose import Pose
from lucida_mini.schema import DatasetManifest
from lucida_mini.state import GizmoState


def pose_from_record(record) -> Pose:
    return Pose(
        np.asarray(record.position_m),
        np.asarray(record.rotation_object_to_world),
        np.asarray(record.size_m),
    )


def replay_iou(trajectory, actions: list[dict[str, str]]) -> float:
    current = GizmoState(pose_from_record(trajectory.initial_pose))
    for item in actions:
        if item["source"] == "model_invalid":
            continue
        current = execute_action(current, parse_action(item["action"]))
    return oriented_box_iou(current.pose, pose_from_record(trajectory.target_pose))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        required=True,
        help="Existing eval-<step> directory containing closed_loop.json.",
    )
    args = parser.parse_args()

    manifest = DatasetManifest.model_validate_json(args.manifest.read_text())
    trajectories = {item.trajectory_id: item for item in manifest.trajectories}
    report_path = args.evaluation_dir / "closed_loop.json"
    report = json.loads(report_path.read_text())
    rollouts = report["rollouts"]
    if not rollouts:
        raise ValueError(f"no rollouts in {report_path}")

    for rollout in rollouts:
        trajectory_id = rollout["trajectory_id"]
        try:
            trajectory = trajectories[trajectory_id]
        except KeyError as exc:
            raise ValueError(f"unknown trajectory in {report_path}: {trajectory_id}") from exc
        rollout["metrics"]["iou_3d"] = replay_iou(trajectory, rollout["actions"])

    fields = list(rollouts[0]["metrics"])
    summary = {
        "step": rollouts[0]["metrics"]["step"],
        **{
            field: float(np.mean([rollout["metrics"][field] for rollout in rollouts]))
            for field in fields
            if field not in {"step", "trajectory_id", "context_id"}
        },
    }
    report["summary"] = summary
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with (args.evaluation_dir / "per_trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rollout["metrics"] for rollout in rollouts)
    with (args.evaluation_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(f"rescored {len(rollouts)} saved rollouts in {args.evaluation_dir}")
    print(f"mean oriented 3D IoU: {summary['iou_3d']:.6f}")


if __name__ == "__main__":
    main()
