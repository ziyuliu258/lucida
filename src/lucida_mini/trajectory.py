from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import PermuteAxis, Stop, SwitchObs, UpdatePose
from .expert import (
    ExpertThresholds,
    PrivilegedExpert,
    all_axis_permutations,
    execute_action,
)
from .pose import Pose, rotation_zxy
from .schema import PoseRecord, TrajectoryRecord, TurnRecord
from .serialization import action_to_text
from .state import GizmoState, ObservationMode


@dataclass(frozen=True)
class PerturbationConfig:
    rotation_deg: float = 100.0
    translation_fraction: float = 0.35
    log_scale: float = 0.30
    inject_error_probability: float = 0.55
    max_steps: int = 24


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


def _copy_rng(rng: np.random.Generator) -> np.random.Generator:
    cloned = np.random.default_rng()
    cloned.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    return cloned


def _observation_paths(base: Path, mode: ObservationMode) -> list[str]:
    """Paths match the separate main views and optional six orthographic views."""
    paths = [base.as_posix()]
    paths.extend(
        base.with_name(f"{base.stem}_{suffix}{base.suffix}").as_posix()
        for suffix in ("overlay", "pointcloud")
    )
    if mode is ObservationMode.SIX_AXIS:
        paths.extend(
            base.with_name(f"{base.stem}_local_{axis}_{sign}{base.suffix}").as_posix()
            for axis in ("x", "y", "z")
            for sign in ("pos", "neg")
        )
    return paths


def _scaled_update(action: UpdatePose, factor: float) -> UpdatePose:
    def scale(values: np.ndarray | None, *, positive_floor: bool = False):
        if values is None:
            return None
        result = np.asarray(values, dtype=np.float64) * factor
        if positive_floor:
            result = np.maximum(result, -0.95)
        return np.round(result, 2)

    return UpdatePose(
        rotate_zxy_deg=scale(action.rotate_zxy_deg),
        translate_local_fraction=scale(action.translate_local_fraction),
        scale_fraction=scale(action.scale_fraction, positive_floor=True),
    )


def _small_wrong_update(rng: np.random.Generator) -> UpdatePose:
    return UpdatePose(
        rotate_zxy_deg=np.round(rng.uniform(-12.0, 12.0, size=3), 2),
        translate_local_fraction=np.round(rng.uniform(-0.04, 0.04, size=3), 2),
    )


def _error_candidates(
    expected: object,
    current: GizmoState,
    target: Pose,
    rng: np.random.Generator,
) -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = []
    if isinstance(expected, UpdatePose):
        candidates.extend(
            [
                ("overcorrection", _scaled_update(expected, 1.5)),
                ("undercorrection", _scaled_update(expected, 0.5)),
            ]
        )
    if isinstance(expected, SwitchObs):
        helper = PrivilegedExpert(target)
        best, _ = helper._best_permutation(current.pose)
        candidates.append(("skipped_view_switch", best))
    if isinstance(expected, PermuteAxis):
        wrong = [item for item in all_axis_permutations() if item != expected]
        candidates.append(("wrong_axis_permutation", wrong[int(rng.integers(len(wrong)))]))
        candidates.append(("skipped_axis_permutation", _small_wrong_update(rng)))
    return candidates


def _recoverable(
    state: GizmoState,
    erroneous_action: object,
    target: Pose,
    thresholds: ExpertThresholds,
    remaining_steps: int,
    seed: int,
) -> bool:
    try:
        current = execute_action(state, erroneous_action)
    except (ValueError, TypeError, FloatingPointError):
        return False
    expert = PrivilegedExpert(target=target, thresholds=thresholds)
    rng = np.random.default_rng(seed)
    for _ in range(remaining_steps):
        action = expert.act(current.pose, rng, current.observation_mode)
        current = execute_action(current, action)
        if isinstance(action, Stop):
            return True
    return False


def generate_expert_trajectory(
    context_id: str,
    index: int,
    seed: int,
    target: Pose,
    observation_dir: Path,
    thresholds: ExpertThresholds,
    config: PerturbationConfig,
    observation_frame_count: int = 1,
) -> TrajectoryRecord:
    if config.max_steps > 64:
        raise ValueError("max_steps must be at most 64")
    if observation_frame_count < 1:
        raise ValueError("observation_frame_count must be positive")
    rng = np.random.default_rng(seed)
    initial = perturb_pose(target, rng, config)
    current = GizmoState(initial)
    expert = PrivilegedExpert(target=target, thresholds=thresholds)
    turns: list[TurnRecord] = []
    injected = False

    for step in range(config.max_steps):
        before = current
        should_try_error = (
            not injected
            and step < config.max_steps - 3
            and rng.random() < config.inject_error_probability
        )
        action = None
        error_type = None
        if should_try_error:
            # Preview with cloned state/RNG; this does not advance the real expert.
            preview_expert = copy.deepcopy(expert)
            preview = preview_expert.act(
                current.pose, _copy_rng(rng), current.observation_mode
            )
            candidates = _error_candidates(preview, current, target, rng)
            if candidates:
                start = int(rng.integers(len(candidates)))
                candidates = candidates[start:] + candidates[:start]
            for candidate_index, (candidate_kind, candidate_action) in enumerate(candidates):
                if _recoverable(
                    current,
                    candidate_action,
                    target,
                    thresholds,
                    config.max_steps - step - 1,
                    seed + 7919 * (step + 1) + candidate_index,
                ):
                    action = candidate_action
                    error_type = candidate_kind
                    break
        use_error = action is not None
        if not use_error:
            action = expert.act(current.pose, rng, current.observation_mode)

        current = execute_action(current, action)
        image_prefix = observation_dir / f"step_{step:02d}.png"
        frame_index = min(step, observation_frame_count - 1)
        turns.append(
            TurnRecord(
                step=step,
                observation_paths=_observation_paths(image_prefix, before.observation_mode),
                observation_frame_index=frame_index,
                observation_mode_before=before.observation_mode,
                observation_mode_after=current.observation_mode,
                action=action_to_text(action),
                injected_error=use_error,
                injected_error_type=error_type,
                supervise=not use_error,
                state_before=pose_to_record(before.pose),
                state_after=pose_to_record(current.pose),
            )
        )
        if use_error:
            injected = True
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
