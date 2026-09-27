from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .pose import Pose


class ObservationMode(StrEnum):
    SCENE = "scene"
    SIX_AXIS = "six_axis"

    def toggled(self) -> "ObservationMode":
        return ObservationMode.SIX_AXIS if self is ObservationMode.SCENE else ObservationMode.SCENE


@dataclass(frozen=True)
class GizmoState:
    pose: Pose
    observation_mode: ObservationMode = ObservationMode.SCENE
