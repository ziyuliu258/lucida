#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lucida_mini.expert import ExpertThresholds
from lucida_mini.pose import Pose
from lucida_mini.render import render_ca1m_observation, render_foundation_observation, render_mesatask_observation
from lucida_mini.trajectory import PerturbationConfig, generate_expert_trajectory


def record_to_pose(record) -> Pose:
    return Pose(
        position=np.asarray(record.position_m),
        rotation=np.asarray(record.rotation_object_to_world),
        size=np.asarray(record.size_m),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("context_id")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    context_dir = args.dataset_root / "contexts" / args.context_id
    context = json.loads((context_dir / "context.json").read_text())
    target = context["target_pose"]
    target_pose = Pose(
        position=np.asarray(target["position_m"]),
        rotation=np.asarray(target["rotation_object_to_world"]),
        size=np.asarray(target["size_m"]),
    )
    render = {
        "foundationpose": render_foundation_observation,
        "populated_3d_front": render_mesatask_observation,
        "ca1m_objects": render_ca1m_observation,
    }.get(context["source"])
    if render is None:
        raise NotImplementedError(f"no renderer for source {context['source']}")

    trajectories = []
    for index in range(10):
        relative_dir = Path("trajectories") / args.context_id / f"{index:02d}"
        trajectory = generate_expert_trajectory(
            context_id=args.context_id,
            index=index,
            seed=args.seed + index,
            target=target_pose,
            observation_dir=relative_dir,
            thresholds=ExpertThresholds(),
            config=PerturbationConfig(),
        )
        actual_dir = args.dataset_root / relative_dir
        actual_dir.mkdir(parents=True, exist_ok=True)
        for turn in trajectory.turns:
            render(
                context_dir,
                record_to_pose(turn.state_before),
                args.dataset_root / turn.observation_paths[0],
                mode=turn.observation_mode_before,
            )
        (actual_dir / "trajectory.json").write_text(trajectory.model_dump_json(indent=2))
        trajectories.append(trajectory)
        print(trajectory.trajectory_id, len(trajectory.turns), trajectory.termination)
    if any(item.termination != "stop" for item in trajectories):
        raise RuntimeError("at least one expert trajectory failed to stop within the budget")


if __name__ == "__main__":
    main()
