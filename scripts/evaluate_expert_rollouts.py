#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh

from lucida_mini.metrics import (
    add_sb,
    object_diameter,
    oriented_box_iou,
    rotation_geodesic_deg,
    transform_normalized_points,
)
from lucida_mini.pose import Pose
from lucida_mini.schema import DatasetManifest


def pose(record) -> Pose:
    return Pose(np.asarray(record.position_m), np.asarray(record.rotation_object_to_world), np.asarray(record.size_m))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--surface-points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    manifest_path = args.manifest or args.dataset_root / "manifest.json"
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text())
    if not manifest.frozen:
        raise ValueError("expert evaluation requires a frozen manifest")
    contexts = {item.context_id: item for item in manifest.contexts}
    points_by_context = {}
    for context_id, context in contexts.items():
        mesh = trimesh.load(args.dataset_root / context.mesh_path, force="mesh", process=False)
        points, _ = trimesh.sample.sample_surface(mesh, args.surface_points, seed=args.seed)
        target_pose = pose(next(
            trajectory.target_pose for trajectory in manifest.trajectories
            if trajectory.context_id == context_id
        ))
        target_points = transform_normalized_points(points, target_pose)
        points_by_context[context_id] = (
            points, target_pose, target_points, object_diameter(target_points)
        )

    rows = []
    for trajectory in manifest.trajectories:
        points, target_pose, target_points, diameter = points_by_context[trajectory.context_id]
        predicted_pose = pose(trajectory.turns[-1].state_after)
        predicted_points = transform_normalized_points(points, predicted_pose)
        distance = add_sb(predicted_points, target_points)
        fraction = distance / diameter if diameter > 0 else float("inf")
        rotation_error = rotation_geodesic_deg(predicted_pose, target_pose)
        metrics = {
            "add_sb_m": distance,
            "object_diameter_m": diameter,
            "add_sb_fraction": fraction,
            "add_sb_at_0.10": float(fraction < 0.10),
            "add_sb_at_0.05": float(fraction < 0.05),
            "add_sb_at_0.01": float(fraction < 0.01),
            "iou_3d": oriented_box_iou(predicted_pose, target_pose),
            "rotation_error_deg": rotation_error,
            "rotation_at_5deg": float(rotation_error < 5.0),
            "closed_loop_success": float(
                trajectory.termination == "stop" and fraction < 0.05 and rotation_error < 5.0
            ),
        }
        rows.append({
            "trajectory_id": trajectory.trajectory_id,
            "context_id": trajectory.context_id,
            "steps": len(trajectory.turns),
            "stopped": float(trajectory.termination == "stop"),
            **metrics,
        })
    if not rows:
        raise ValueError("no trajectories found")
    metric_keys = [key for key in rows[0] if key not in {"trajectory_id", "context_id"}]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    summary["trajectories"] = len(rows)
    payload = {"summary": summary, "per_trajectory": rows}
    output = args.output or args.dataset_root / "expert_metrics.json"
    output.write_text(json.dumps(payload, indent=2))
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
