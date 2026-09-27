from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from .actions import PermuteAxis, Stop, SwitchObs, UpdatePose
from .metrics import rotation_geodesic_deg
from .pose import Pose
from .state import GizmoState, ObservationMode


def all_axis_permutations() -> tuple[PermuteAxis, ...]:
    names = ("x", "-x", "y", "-y", "z", "-z")
    actions: list[PermuteAxis] = []
    for x, z in product(names, repeat=2):
        action = PermuteAxis(x=x, z=z)
        try:
            action.matrix()
        except ValueError:
            continue
        actions.append(action)
    if len(actions) != 24:
        raise AssertionError("expected exactly 24 right-handed signed-axis permutations")
    return tuple(actions)


@dataclass(frozen=True)
class ExpertThresholds:
    rotation_stop_deg: float = 0.5
    translation_stop_fraction: float = 0.005
    log_scale_stop: float = 0.005
    coarse_rotation_min_deg: float = 45.0
    decimals: int = 2


@dataclass
class PrivilegedExpert:
    target: Pose
    thresholds: ExpertThresholds = ExpertThresholds()
    coarse_view_requested: bool = False

    def _rotation_error(self, current: Pose) -> float:
        return rotation_geodesic_deg(current, self.target)

    def _best_permutation(self, current: Pose) -> tuple[PermuteAxis, float]:
        candidates = [
            (action, self._rotation_error(action.apply(current)))
            for action in all_axis_permutations()
        ]
        return min(candidates, key=lambda item: item[1])

    def act(self, current: Pose):
        rotation_error = self._rotation_error(current)
        normalized_translation = np.linalg.norm(
            (self.target.position - current.position) / self.target.size
        )
        log_scale_error = np.max(np.abs(np.log(self.target.size / current.size)))

        if (
            rotation_error <= self.thresholds.rotation_stop_deg
            and normalized_translation <= self.thresholds.translation_stop_fraction
            and log_scale_error <= self.thresholds.log_scale_stop
        ):
            return Stop()

        if rotation_error >= self.thresholds.coarse_rotation_min_deg:
            permutation, residual = self._best_permutation(current)
            improvement = rotation_error - residual
            if improvement >= self.thresholds.coarse_rotation_min_deg / 2:
                if not self.coarse_view_requested:
                    self.coarse_view_requested = True
                    return SwitchObs()
                self.coarse_view_requested = False
                return permutation

        relative = current.rotation.T @ self.target.rotation
        # Uppercase means intrinsic rotations, matching Rz @ Rx @ Ry in pose.rotation_zxy.
        delta_r = Rotation.from_matrix(relative).as_euler("ZXY", degrees=True)
        next_rotation = current.rotation @ Rotation.from_euler(
            "ZXY", delta_r, degrees=True
        ).as_matrix()
        delta_p = (next_rotation.T @ (self.target.position - current.position)) / current.size
        delta_s = self.target.size / current.size - 1.0

        def rounded_or_none(values: np.ndarray, active: bool) -> np.ndarray | None:
            return np.round(values, self.thresholds.decimals) if active else None

        rotation_update = rounded_or_none(
            delta_r, rotation_error > self.thresholds.rotation_stop_deg
        )
        translation_update = rounded_or_none(
            delta_p, normalized_translation > self.thresholds.translation_stop_fraction
        )
        scale_update = rounded_or_none(
            delta_s, log_scale_error > self.thresholds.log_scale_stop
        )

        # The text protocol is limited to ``decimals`` places. Once a residual
        # quantizes entirely to zero it cannot be improved without oscillation.
        def omit_zero(values: np.ndarray | None) -> np.ndarray | None:
            return None if values is not None and not np.any(values) else values

        rotation_update = omit_zero(rotation_update)
        translation_update = omit_zero(translation_update)
        scale_update = omit_zero(scale_update)
        if rotation_update is None and translation_update is None and scale_update is None:
            return Stop()
        return UpdatePose(
            rotate_zxy_deg=rotation_update,
            translate_local_fraction=translation_update,
            scale_fraction=scale_update,
        )


def execute_action(state: GizmoState | Pose, action) -> GizmoState:
    """Apply pose actions and observation switches through one environment API.

    A Pose argument is accepted for geometry-only callers and starts in scene
    mode. The returned value is always a GizmoState so callers cannot silently
    discard a switch_obs transition.
    """
    if isinstance(state, Pose):
        state = GizmoState(state)
    if isinstance(action, (UpdatePose, PermuteAxis)):
        return GizmoState(action.apply(state.pose), state.observation_mode)
    if isinstance(action, SwitchObs):
        return GizmoState(state.pose, state.observation_mode.toggled())
    return state
