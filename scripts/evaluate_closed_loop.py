#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from lucida_mini.actions import Stop
from lucida_mini.expert import execute_action
from lucida_mini.evaluation import full_loop_success, iter_rollout_schedule, rollout_protocol
from lucida_mini.metrics import (
    add_sb,
    object_diameter,
    oriented_box_iou,
    rotation_geodesic_deg,
    transform_normalized_points,
)
from lucida_mini.pose import Pose
from lucida_mini.render import (
    render_ca1m_observation,
    render_foundation_observation,
    render_mesatask_observation,
)
from lucida_mini.schema import DatasetManifest
from lucida_mini.actions import parse_action
from lucida_mini.prompts import SYSTEM_PROMPT, frame_index_for_turn, instruction_for_turn
from lucida_mini.serialization import action_to_text
from lucida_mini.state import GizmoState
from lucida_mini.vision import (
    image_paths_in_messages,
    validate_image_files,
    validate_processed_image_grids,
)


def to_pose(record) -> Pose:
    return Pose(
        np.asarray(record.position_m),
        np.asarray(record.rotation_object_to_world),
        np.asarray(record.size_m),
    )


def generate_action(
    model,
    processor,
    messages: list[dict],
    max_new_tokens: int,
    max_sequence_length: int,
    vision_max_pixels: int,
) -> str:
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_paths = image_paths_in_messages(messages)
    image_processor = processor.image_processor
    patch_size = int(getattr(image_processor, "patch_size", 16))
    merge_size = int(getattr(image_processor, "spatial_merge_size", 2))
    image_sizes = validate_image_files(
        image_paths,
        vision_max_pixels,
        patch_size=patch_size,
        spatial_merge_size=merge_size,
    )
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[text], images=images, videos=videos, padding=False, return_tensors="pt"
    )
    validate_processed_image_grids(inputs, image_sizes, patch_size=patch_size)
    if inputs["input_ids"].shape[1] + max_new_tokens > max_sequence_length:
        raise ValueError(
            f"rollout prompt plus generation reaches max_sequence_length={max_sequence_length}; "
            "refusing to discard prior observations"
        )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generated = output[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--adapter", type=Path,
        help="Optional LoRA adapter; omit to evaluate the pretrained base as-is.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-sequence-length", type=int, default=131072)
    parser.add_argument("--surface-points", type=int, default=10000)
    parser.add_argument("--vision-max-pixels", type=int, default=786432)
    parser.add_argument("--vision-min-pixels", type=int, default=65536)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard-count)")

    manifest = DatasetManifest.model_validate_json(args.manifest.read_text())
    missing = manifest.validate_files(args.dataset_root)
    if missing:
        raise FileNotFoundError(
            f"manifest references {len(missing)} missing files; first: {missing[0]}"
        )
    manifest.validate_corrected_inputs(args.dataset_root)
    contexts = {item.context_id: item for item in manifest.contexts}
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.image_processor.max_pixels = args.vision_max_pixels
    processor.image_processor.min_pixels = args.vision_min_pixels
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = (
        PeftModel.from_pretrained(base, args.adapter)
        if args.adapter is not None else base
    ).eval().cuda()

    renderers = {
        "populated_3d_front": render_mesatask_observation,
        "foundationpose": render_foundation_observation,
        "ca1m_objects": render_ca1m_observation,
    }
    geometry: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for context_id, context in contexts.items():
        mesh = trimesh.load(args.dataset_root / context.mesh_path, force="mesh", process=False)
        points, _ = trimesh.sample.sample_surface(mesh, args.surface_points, seed=20260922)
        target = to_pose(context.target_pose)
        target_points = transform_normalized_points(points, target)
        geometry[context_id] = points, target_points, object_diameter(target_points)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rollouts: list[dict] = []
    for trajectory_index, trajectory in enumerate(manifest.trajectories):
        if trajectory_index % args.shard_count != args.shard_index:
            continue
        context = contexts[trajectory.context_id]
        context_dir = args.dataset_root / "contexts" / trajectory.context_id
        renderer = renderers[context.source]
        current = GizmoState(to_pose(trajectory.initial_pose))
        target = to_pose(trajectory.target_pose)
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        valid_actions = 0
        model_actions = 0
        injected_actions = 0
        stopped = False
        actions: list[dict[str, str]] = []
        termination = "max_steps"
        rollout_dir = args.output_dir / "rollouts" / trajectory.trajectory_id

        # Replay injected, recoverable environment errors as masked history;
        # they remain visible to the model but are not predicted as labels.
        for turn_index, injected_action in iter_rollout_schedule(trajectory, args.max_steps):
            image_path = rollout_dir / f"step_{turn_index:02d}.png"
            turn_context = {"step": turn_index}
            frame_index = frame_index_for_turn(context, turn_context, args.dataset_root)
            image_paths = renderer(
                context_dir,
                current.pose,
                image_path,
                mode=current.observation_mode,
                frame_index=frame_index or 0,
            )
            messages.append({
                "role": "user",
                "content": [
                    *({"type": "image", "image": str(path)} for path in image_paths),
                    {"type": "text", "text": instruction_for_turn(context, turn_context, args.dataset_root)},
                ],
            })
            if injected_action is not None:
                action = parse_action(injected_action)
                actions.append({"source": "injected_error", "action": injected_action})
                messages.append({"role": "assistant", "content": injected_action})
                current = execute_action(current, action)
                injected_actions += 1
                continue

            model_actions += 1
            raw = generate_action(
                model,
                processor,
                messages,
                args.max_new_tokens,
                args.max_sequence_length,
                args.vision_max_pixels,
            )
            try:
                action = parse_action(raw)
                canonical = action_to_text(action)
            except (ValueError, TypeError):
                actions.append({"source": "model_invalid", "action": raw})
                termination = "invalid_action"
                break
            try:
                next_state = execute_action(current, action)
            except (ValueError, TypeError, FloatingPointError) as exc:
                actions.append({
                    "source": "model_invalid_environment_action",
                    "action": canonical,
                    "error": str(exc),
                })
                termination = "invalid_action"
                break
            valid_actions += 1
            actions.append({"source": "model", "action": canonical})
            messages.append({"role": "assistant", "content": canonical})
            current = next_state
            if isinstance(action, Stop):
                stopped = True
                termination = "stop"
                break

        points, target_points, diameter = geometry[trajectory.context_id]
        predicted_points = transform_normalized_points(points, current.pose)
        distance = add_sb(predicted_points, target_points)
        fraction = distance / diameter
        rotation_error = rotation_geodesic_deg(current.pose, target)
        row = {
            "step": args.step,
            "trajectory_id": trajectory.trajectory_id,
            "context_id": trajectory.context_id,
            "actions": model_actions,
            "injected_actions": injected_actions,
            "valid_action_rate": valid_actions / max(model_actions, 1),
            "stopped": float(stopped),
            "add_sb_m": distance,
            "object_diameter_m": diameter,
            "add_sb_fraction": fraction,
            "add_sb_at_0.10": float(fraction < 0.10),
            "add_sb_at_0.05": float(fraction < 0.05),
            "add_sb_at_0.01": float(fraction < 0.01),
            "iou_3d": oriented_box_iou(current.pose, target),
            "rotation_error_deg": rotation_error,
            "rotation_at_5deg": float(rotation_error < 5.0),
        }
        row["closed_loop_success"] = full_loop_success(row)
        rows.append(row)
        rollouts.append({
            "trajectory_id": trajectory.trajectory_id,
            "termination": termination,
            "actions": actions,
            "metrics": row,
        })
        print(
            trajectory.trajectory_id,
            termination,
            f"model_actions={model_actions}",
            f"injected_actions={injected_actions}",
            f"ADD-SB={fraction:.4f}",
        )

    keys = [key for key in rows[0] if key not in {"step", "trajectory_id", "context_id"}]
    summary = {"step": args.step, **{key: float(np.mean([row[key] for row in rows])) for key in keys}}
    (args.output_dir / "closed_loop.json").write_text(
        json.dumps({
            "protocol": rollout_protocol(args.max_steps),
            "summary": summary,
            "rollouts": rollouts,
        }, indent=2)
    )
    with (args.output_dir / "per_trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
