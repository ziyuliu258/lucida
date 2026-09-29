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

    @staticmethod
    def _round(values: np.ndarray, decimals: int) -> np.ndarray | None:
        rounded = np.round(values, decimals)
        return None if not np.any(rounded) else rounded

    @staticmethod
    def _partial(values: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
        active = np.flatnonzero(np.abs(values) > 1e-10)
        if not len(active):
            return None
        count = int(rng.integers(1, len(active) + 1))
        selected = rng.choice(active, size=count, replace=False)
        result = np.zeros(3, dtype=np.float64)
        result[selected] = values[selected]
        return result

    def act(
        self,
        current: Pose,
        rng: np.random.Generator | None = None,
        observation_mode: ObservationMode = ObservationMode.SCENE,
    ):
        if rng is None:
            rng = np.random.default_rng(0)
        rotation_error = self._rotation_error(current)
        target_diagonal = float(np.linalg.norm(self.target.size))
        normalized_translation = float(
            np.linalg.norm(self.target.position - current.position) / target_diagonal
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
                if not self.coarse_view_requested and observation_mode is ObservationMode.SCENE:
                    self.coarse_view_requested = True
                    return SwitchObs()
                self.coarse_view_requested = False
                return permutation

        if rotation_error > self.thresholds.rotation_stop_deg:
            relative = current.rotation.T @ self.target.rotation
            # Intrinsic Z-X-Y matches Lucida's GizmoAct rotation convention.
            delta_r = Rotation.from_matrix(relative).as_euler("ZXY", degrees=True)
            rotation_update = self._round(
                self._partial(delta_r, rng) if delta_r is not None else delta_r,
                self.thresholds.decimals,
            )
            if rotation_update is None:
                return Stop()
            return UpdatePose(rotate_zxy_deg=rotation_update)

        local_delta = current.rotation.T @ (self.target.position - current.position)
        delta_p = local_delta / current.size
        delta_s = self.target.size / current.size - 1.0
        translation_active = normalized_translation > self.thresholds.translation_stop_fraction
        scale_active = log_scale_error > self.thresholds.log_scale_stop
        if translation_active:
            selected_translation = self._partial(delta_p, rng)
            delta_p = (
                selected_translation
                if selected_translation is not None
                else np.zeros(3, dtype=np.float64)
            )
        else:
            delta_p = np.zeros(3, dtype=np.float64)
        if scale_active:
            selected_scale = self._partial(delta_s, rng)
            delta_s = (
                selected_scale
                if selected_scale is not None
                else np.zeros(3, dtype=np.float64)
            )
        else:
            delta_s = np.zeros(3, dtype=np.float64)

        translation_update = self._round(delta_p, self.thresholds.decimals)
        scale_update = self._round(delta_s, self.thresholds.decimals)
        if translation_update is None and scale_update is None:
            return Stop()
        return UpdatePose(
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
