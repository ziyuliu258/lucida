#!/usr/bin/env python3
"""Export fixed-set closed-loop action traces and readable contact sheets."""
from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from lucida_mini.actions import parse_action
from lucida_mini.expert import execute_action
from lucida_mini.metrics import (
    add_sb,
    object_diameter,
    oriented_box_iou,
    rotation_geodesic_deg,
    transform_normalized_points,
)
from lucida_mini.pose import Pose
from lucida_mini.schema import DatasetManifest
from lucida_mini.state import GizmoState


def to_pose(record) -> Pose:
    return Pose(
        np.asarray(record.position_m),
        np.asarray(record.rotation_object_to_world),
        np.asarray(record.size_m),
    )


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def find_rollout_dir(eval_dir: Path, trajectory_id: str) -> Path:
    matches = []
    for shard_csv in sorted(eval_dir.glob("shard_*/per_trajectory.csv")):
        with shard_csv.open(newline="") as handle:
            if any(row["trajectory_id"] == trajectory_id for row in csv.DictReader(handle)):
                matches.append(shard_csv.parent / "rollouts" / trajectory_id)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one observation directory for {trajectory_id}, found {matches}"
        )
    return matches[0]


def export_contact_sheet(
    trajectory_id: str,
    context_id: str,
    final_metrics: dict[str, str],
    actions: list[dict],
    rollout_dir: Path,
    output: Path,
) -> None:
    # Native observations write three files for one action.  Keep one preview
    # per action so captions cannot drift to the next observation.
    images_by_step: dict[int, Path] = {}
    for path in sorted(rollout_dir.glob("step_*.png")):
        step = int(path.stem.split("_")[1])
        if step not in images_by_step or path.stem.endswith("_focus"):
            images_by_step[step] = path
    images = [images_by_step[step] for step in sorted(images_by_step)]
    if not images:
        raise FileNotFoundError(f"no observation frames under {rollout_dir}")
    columns = 3
    tile_width, image_height, label_height = 420, 330, 112
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * tile_width, 54 + rows * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(20)
    label_font = load_font(14)
    title = (
        f"{trajectory_id} / {context_id} | success={final_metrics['closed_loop_success']} | "
        f"ADD-SB/d={float(final_metrics['add_sb_fraction']):.4f} | "
        f"rot={float(final_metrics['rotation_error_deg']):.1f} deg"
    )
    draw.text((12, 12), title, fill="black", font=title_font)

    for index, path in enumerate(images):
        action = actions[index] if index < len(actions) else {"source": "no action record", "action": ""}
        x = (index % columns) * tile_width
        y = 54 + (index // columns) * (image_height + label_height)
        with Image.open(path) as source:
            frame = source.convert("RGB")
            frame.thumbnail((tile_width - 12, image_height - 8))
        canvas.paste(frame, (x + (tile_width - frame.width) // 2, y + 4))
        description = f"{index:02d} {action.get('source', '?')}: {action.get('action', '')}"
        if action.get("error"):
            description += f" | {action['error']}"
        wrapped = "\n".join(textwrap.wrap(description, width=53, break_long_words=True))
        draw.multiline_text((x + 8, y + image_height), wrapped, fill="black", font=label_font, spacing=3)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-points", type=int, default=10000)
    parser.add_argument("--limit-trajectories", type=int, default=0, help="Optional small smoke-test limit.")
    args = parser.parse_args()

    manifest = DatasetManifest.model_validate_json(args.manifest.read_text())
    contexts = {item.context_id: item for item in manifest.contexts}
    trajectories = {item.trajectory_id: item for item in manifest.trajectories}
    payload = json.loads((args.eval_dir / "closed_loop.json").read_text())
    with (args.eval_dir / "per_trajectory.csv").open(newline="") as handle:
        final_rows = {row["trajectory_id"]: row for row in csv.DictReader(handle)}

    points_by_context: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for context_id, context in contexts.items():
        mesh = trimesh.load(args.dataset_root / context.mesh_path, force="mesh", process=False)
        points, _ = trimesh.sample.sample_surface(mesh, args.surface_points, seed=20260922)
        target = to_pose(context.target_pose)
        target_points = transform_normalized_points(points, target)
        points_by_context[context_id] = points, target_points, object_diameter(target_points)

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    trace_rows: list[dict] = []
    representatives: dict[str, str] = {}
    failures = []
    rollouts = payload["rollouts"]
    if args.limit_trajectories > 0:
        rollouts = rollouts[:args.limit_trajectories]
    for rollout in rollouts:
        trajectory_id = rollout["trajectory_id"]
        trajectory = trajectories[trajectory_id]
        context = contexts[trajectory.context_id]
        final = final_rows[trajectory_id]
        action_records = rollout["actions"]
        current = GizmoState(to_pose(trajectory.initial_pose))
        target = to_pose(trajectory.target_pose)
        points, target_points, diameter = points_by_context[trajectory.context_id]
        for action_index, record in enumerate(action_records):
            applied = False
            error = record.get("error", "")
            if record["source"] in {"model", "injected_error"}:
                try:
                    current = execute_action(current, parse_action(record["action"]))
                    applied = True
                except (ValueError, TypeError, FloatingPointError) as exc:
                    error = str(exc)
            predicted = transform_normalized_points(points, current.pose)
            distance = add_sb(predicted, target_points)
            rotation_error = rotation_geodesic_deg(current.pose, target)
            trace_rows.append({
                "step": int(final["step"]),
                "trajectory_id": trajectory_id,
                "context_id": trajectory.context_id,
                "action_index": action_index,
                "source": record["source"],
                "action": record["action"],
                "applied": int(applied),
                "error": error,
                "add_sb_m": distance,
                "add_sb_fraction": distance / diameter,
                "iou_3d": oriented_box_iou(current.pose, target),
                "rotation_error_deg": rotation_error,
            })

        rollout_dir = find_rollout_dir(args.eval_dir, trajectory_id)
        if trajectory.context_id not in representatives:
            representatives[trajectory.context_id] = trajectory_id
        if float(final["closed_loop_success"]) < 1.0:
            failures.append(trajectory_id)
            export_contact_sheet(
                trajectory_id,
                trajectory.context_id,
                final,
                action_records,
                rollout_dir,
                output_root / "failures" / f"{trajectory_id}.png",
            )

    fields = list(trace_rows[0]) if trace_rows else []
    with (output_root / "step_traces.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(trace_rows)

    index_rows = []
    for context_id, trajectory_id in sorted(representatives.items()):
        final = final_rows[trajectory_id]
        export_contact_sheet(
            trajectory_id,
            context_id,
            final,
            next(item["actions"] for item in payload["rollouts"] if item["trajectory_id"] == trajectory_id),
            find_rollout_dir(args.eval_dir, trajectory_id),
            output_root / "representatives" / f"{context_id}.png",
        )
        index_rows.append({
            "context_id": context_id,
            "trajectory_id": trajectory_id,
            "closed_loop_success": final["closed_loop_success"],
            "contact_sheet": f"representatives/{context_id}.png",
        })
    with (output_root / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"exported {len(trace_rows)} action-state rows, {len(index_rows)} context examples, {len(failures)} failure sheets")


if __name__ == "__main__":
    main()
