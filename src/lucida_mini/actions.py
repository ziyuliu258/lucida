from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .pose import Pose

_ACTION_RE = re.compile(r"^<(update_pose|stop|switch_obs|permute_axis)>(.*)</\1>$", re.DOTALL)
_AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


@dataclass(frozen=True)
class UpdatePose:
    rotate_zxy_deg: np.ndarray | None = None
    translate_local_fraction: np.ndarray | None = None
    scale_fraction: np.ndarray | None = None

    def apply(self, pose: Pose) -> Pose:
        return pose.update(
            rotate_zxy_deg=self.rotate_zxy_deg,
            translate_local_fraction=self.translate_local_fraction,
            scale_fraction=self.scale_fraction,
        )


@dataclass(frozen=True)
class Stop:
    pass


@dataclass(frozen=True)
class SwitchObs:
    mode: str = "permute_axis"


@dataclass(frozen=True)
class PermuteAxis:
    x: str
    z: str

    def matrix(self) -> np.ndarray:
        ex, ez = _AXES[self.x], _AXES[self.z]
        if not np.isclose(ex @ ez, 0.0):
            raise ValueError("permuted x and z axes must be orthogonal")
        matrix = np.column_stack((ex, np.cross(ez, ex), ez))
        if not np.isclose(np.linalg.det(matrix), 1.0):
            raise ValueError("axis permutation must be right-handed")
        return matrix

    def apply(self, pose: Pose) -> Pose:
        return pose.permute(self.matrix())


Action: TypeAlias = UpdatePose | Stop | SwitchObs | PermuteAxis


def _vector(payload: dict, key: str, fields: tuple[str, str, str]) -> np.ndarray | None:
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, dict) or not all(field in value for field in fields):
        raise ValueError(f"{key} must contain {fields}")
    result = np.asarray([value[field] for field in fields], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{key} values must be finite")
    return result


def parse_action(text: str) -> Action:
    match = _ACTION_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError("action must be exactly one supported XML tag")
    name, body = match.groups()
    if name == "switch_obs":
        if body.strip() != "permute_axis":
            raise ValueError("switch_obs body must be permute_axis")
        return SwitchObs()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("action body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("action payload must be an object")
    if name == "stop":
        if payload:
            raise ValueError("stop payload must be empty")
        return Stop()
    if name == "permute_axis":
        if set(payload) != {"x", "z"} or payload["x"] not in _AXES or payload["z"] not in _AXES:
            raise ValueError("permute_axis requires valid signed x and z axes")
        action = PermuteAxis(payload["x"], payload["z"])
        action.matrix()
        return action

    allowed = {"rotate", "translate", "scale"}
    if not payload or not set(payload).issubset(allowed):
        raise ValueError("update_pose requires a non-empty subset of rotate/translate/scale")
    if "rotate" in payload:
        rotate = payload["rotate"]
        if not isinstance(rotate, dict) or rotate.get("order") != "ZXY":
            raise ValueError("rotation order must be ZXY")
    return UpdatePose(
        rotate_zxy_deg=_vector(payload, "rotate", ("z", "x", "y")),
        translate_local_fraction=_vector(payload, "translate", ("x", "y", "z")),
        scale_fraction=_vector(payload, "scale", ("x", "y", "z")),
    )
