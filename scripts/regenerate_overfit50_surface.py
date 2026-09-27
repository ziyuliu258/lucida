#!/usr/bin/env python3
"""Regenerate only the observations for the already-frozen 50 trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from lucida_mini.actions import parse_action
from lucida_mini.expert import execute_action
from lucida_mini.pose import Pose
from lucida_mini.render import (
    render_ca1m_observation,
    render_foundation_observation,
    render_mesatask_observation,
    render_native_focus_observation,
)
from lucida_mini.schema import DatasetManifest, PoseRecord, TrajectoryRecord, TurnRecord
from lucida_mini.state import GizmoState, ObservationMode


RENDERERS = {
    "foundationpose": render_foundation_observation,
    "populated_3d_front": render_mesatask_observation,
    "ca1m_objects": render_ca1m_observation,
}


def as_pose(record: PoseRecord) -> Pose:
    return Pose(
        np.asarray(record.position_m, dtype=np.float64),
        np.asarray(record.rotation_object_to_world, dtype=np.float64),
        np.asarray(record.size_m, dtype=np.float64),
    )


def same_pose(first: Pose, second: Pose, atol: float = 1e-10) -> bool:
    return all(
        np.allclose(left, right, atol=atol, rtol=0)
        for left, right in (
            (first.position, second.position),
            (first.rotation, second.rotation),
            (first.size, second.size),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--native-focus",
        action="store_true",
        help="Preserve native scene RGB and mesh images and add a shared target crop.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output_root}")
    source_manifest_path = source_root / "manifest.json"
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest = DatasetManifest.model_validate_json(source_manifest_bytes)
    if not source_manifest.frozen:
        raise ValueError("source manifest must be frozen")

    output_root.mkdir(parents=True)
    (output_root / "trajectories").mkdir()
    # The data observations and manifest are new; immutable context inputs stay
    # shared with the frozen source dataset to avoid duplicating large meshes.
    os.symlink(
        os.path.relpath(source_root / "contexts", output_root),
        output_root / "contexts",
        target_is_directory=True,
    )

    context_by_id = {item.context_id: item for item in source_manifest.contexts}
    new_trajectories: list[TrajectoryRecord] = []
    action_count = 0
    switch_count = 0
    for trajectory_index, old_trajectory in enumerate(source_manifest.trajectories):
        context = context_by_id[old_trajectory.context_id]
        context_dir = source_root / "contexts" / old_trajectory.context_id
        render = RENDERERS[context.source]
        state = GizmoState(as_pose(old_trajectory.initial_pose))
        new_turns: list[TurnRecord] = []
        trajectory_index_in_context = int(old_trajectory.trajectory_id.rsplit("_", 1)[-1])
        output_dir = output_root / "trajectories" / old_trajectory.context_id / f"{trajectory_index_in_context:02d}"
        output_dir.mkdir(parents=True)

        for old_turn in old_trajectory.turns:
            if not same_pose(state.pose, as_pose(old_turn.state_before)):
                raise ValueError(
                    f"state_before mismatch while replaying {old_trajectory.trajectory_id} step {old_turn.step}"
                )
            image_rel = Path("trajectories") / old_trajectory.context_id / f"{trajectory_index_in_context:02d}" / f"step_{old_turn.step:02d}.png"
            if args.native_focus:
                rendered_paths = render_native_focus_observation(
                    context_dir, state.pose, output_root / image_rel,
                    mode=state.observation_mode,
                )
            else:
                render(
                    context_dir, state.pose, output_root / image_rel,
                    mode=state.observation_mode,
                )
                rendered_paths = [output_root / image_rel]
            before_mode = state.observation_mode
            action = parse_action(old_turn.action)
            state = execute_action(state, action)
            if not same_pose(state.pose, as_pose(old_turn.state_after)):
                raise ValueError(
                    f"state_after mismatch while replaying {old_trajectory.trajectory_id} step {old_turn.step}"
                )
            switch_count += int(before_mode != state.observation_mode)
            action_count += 1
            new_turns.append(
                TurnRecord(
                    step=old_turn.step,
                    observation_paths=[
                        path.relative_to(output_root).as_posix() for path in rendered_paths
                    ],
                    observation_mode_before=before_mode,
                    observation_mode_after=state.observation_mode,
                    action=old_turn.action,
                    injected_error=old_turn.injected_error,
                    supervise=old_turn.supervise,
                    state_before=old_turn.state_before,
                    state_after=old_turn.state_after,
                )
            )

        new_trajectories.append(
            old_trajectory.model_copy(update={"turns": new_turns}, deep=True)
        )
        print(
            f"[{trajectory_index + 1}/50] {old_trajectory.trajectory_id} "
            f"turns={len(new_turns)} switches="
            f"{sum(t.action.startswith('<switch_obs>') for t in new_turns)}"
        )

    regenerated = DatasetManifest(
        name=source_manifest.name,
        frozen=True,
        contexts=source_manifest.contexts,
        trajectories=new_trajectories,
    )
    missing = regenerated.validate_files(output_root)
    if missing:
        raise FileNotFoundError(f"generated manifest has {len(missing)} missing files; first: {missing[0]}")
    (output_root / "manifest.json").write_text(regenerated.model_dump_json(indent=2) + "\n")
    supervised = sum(turn.supervise for traj in regenerated.trajectories for turn in traj.turns)
    injected = sum(turn.injected_error for traj in regenerated.trajectories for turn in traj.turns)
    report = {
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "frozen_contexts": len(regenerated.contexts),
        "frozen_trajectories": len(regenerated.trajectories),
        "turns": action_count,
        "supervised_turns": supervised,
        "injected_error_turns": injected,
        "switch_obs_turns": switch_count,
        "pose_and_action_replay_exact": True,
        "observation_modes": [mode.value for mode in ObservationMode],
        "observation_layout": "native_rgb_mesh_and_focus_crop" if args.native_focus else "contact_sheet",
    }
    (output_root / "regeneration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
