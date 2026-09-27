from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import Stop, UpdatePose
from .expert import ExpertThresholds, PrivilegedExpert, execute_action
from .pose import Pose, rotation_zxy
from .schema import PoseRecord, TrajectoryRecord, TurnRecord
from .serialization import action_to_text
from .state import GizmoState


@dataclass(frozen=True)
class PerturbationConfig:
    rotation_deg: float = 100.0
    translation_fraction: float = 0.35
    log_scale: float = 0.30
    inject_error_probability: float = 0.30
    max_steps: int = 12


def pose_to_record(pose: Pose) -> PoseRecord:
    return PoseRecord(
        position_m=pose.position.tolist(),
        rotation_object_to_world=pose.rotation.tolist(),
        size_m=pose.size.tolist(),
    )


def perturb_pose(target: Pose, rng: np.random.Generator, config: PerturbationConfig) -> Pose:
    angles = rng.uniform(-config.rotation_deg, config.rotation_deg, size=3)
    translation = rng.uniform(
        -config.translation_fraction, config.translation_fraction, size=3
    )
    scale = np.exp(rng.uniform(-config.log_scale, config.log_scale, size=3))
    return Pose(
        position=target.position + target.rotation @ (translation * target.size),
        rotation=target.rotation @ rotation_zxy(angles),
        size=target.size * scale,
    )


def generate_expert_trajectory(
    context_id: str,
    index: int,
    seed: int,
    target: Pose,
    observation_dir: Path,
    thresholds: ExpertThresholds,
    config: PerturbationConfig,
) -> TrajectoryRecord:
    rng = np.random.default_rng(seed)
    initial = perturb_pose(target, rng, config)
    current = GizmoState(initial)
    expert = PrivilegedExpert(target=target, thresholds=thresholds)
    turns: list[TurnRecord] = []
    injected = False

    for step in range(config.max_steps):
        before = current
        use_error = (
            not injected
            and step < config.max_steps - 3
            and rng.random() < config.inject_error_probability
        )
        if use_error:
            # Small, recoverable local perturbation. It remains in history but is masked.
            action = UpdatePose(
                rotate_zxy_deg=np.round(rng.uniform(-8.0, 8.0, size=3), 2),
                translate_local_fraction=np.round(rng.uniform(-0.03, 0.03, size=3), 2),
            )
            injected = True
        else:
            action = expert.act(current.pose)
        current = execute_action(current, action)
        observation = observation_dir / f"step_{step:02d}.png"
        turns.append(
            TurnRecord(
                step=step,
                observation_paths=[str(observation)],
                observation_mode_before=before.observation_mode,
                observation_mode_after=current.observation_mode,
                action=action_to_text(action),
                injected_error=use_error,
                supervise=not use_error,
                state_before=pose_to_record(before.pose),
                state_after=pose_to_record(current.pose),
            )
        )
        if isinstance(action, Stop):
            return TrajectoryRecord(
                trajectory_id=f"{context_id}_{index:02d}",
                context_id=context_id,
                seed=seed,
                initial_pose=pose_to_record(initial),
                target_pose=pose_to_record(target),
                turns=turns,
                termination="stop",
            )

    return TrajectoryRecord(
        trajectory_id=f"{context_id}_{index:02d}",
        context_id=context_id,
        seed=seed,
        initial_pose=pose_to_record(initial),
        target_pose=pose_to_record(target),
        turns=turns,
        termination="max_steps",
    )
