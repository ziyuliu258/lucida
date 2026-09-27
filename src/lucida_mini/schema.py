from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from .state import ObservationMode


class PoseRecord(BaseModel):
    position_m: list[float] = Field(min_length=3, max_length=3)
    rotation_object_to_world: list[list[float]]
    size_m: list[float] = Field(min_length=3, max_length=3)

    @field_validator("rotation_object_to_world")
    @classmethod
    def validate_rotation(cls, value: list[list[float]]) -> list[list[float]]:
        rotation = np.asarray(value, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("rotation must be right-handed")
        return value

    @field_validator("size_m")
    @classmethod
    def validate_size(cls, value: list[float]) -> list[float]:
        if any(component <= 0 for component in value):
            raise ValueError("size components must be positive")
        return value


class TurnRecord(BaseModel):
    step: int = Field(ge=0, le=11)
    observation_paths: list[str] = Field(min_length=1)
    observation_mode_before: ObservationMode = ObservationMode.SCENE
    observation_mode_after: ObservationMode = ObservationMode.SCENE
    action: str
    injected_error: bool = False
    supervise: bool = True
    state_before: PoseRecord
    state_after: PoseRecord

    @model_validator(mode="after")
    def injected_errors_are_masked(self) -> "TurnRecord":
        if self.injected_error and self.supervise:
            raise ValueError("injected error actions must not be supervised")
        if self.action.startswith("<switch_obs>") and self.state_before != self.state_after:
            raise ValueError("switch_obs must leave the object pose unchanged")
        return self


class TrajectoryRecord(BaseModel):
    trajectory_id: str
    context_id: str
    seed: int
    initial_pose: PoseRecord
    target_pose: PoseRecord
    turns: list[TurnRecord] = Field(min_length=1, max_length=12)
    termination: Literal["stop", "max_steps", "invalid_action"]


class ContextRecord(BaseModel):
    context_id: str
    source: Literal["populated_3d_front", "foundationpose", "ca1m_objects"]
    source_identifier: str
    target_description: str
    target_category: str
    target_bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    mesh_path: str
    rgb_paths: list[str] = Field(min_length=1, max_length=4)
    depth_paths: list[str] = Field(default_factory=list, max_length=4)
    camera_path: str
    point_cloud_path: str
    target_pose: PoseRecord
    coordinate_convention: str
    units: Literal["meters"] = "meters"


class DatasetManifest(BaseModel):
    name: Literal["gizmoact_overfit50"]
    frozen: bool
    contexts: list[ContextRecord]
    trajectories: list[TrajectoryRecord]

    @model_validator(mode="after")
    def validate_fixed_50_contract(self) -> "DatasetManifest":
        expected = {
            "front_01": "populated_3d_front",
            "front_02": "populated_3d_front",
            "foundationpose_01": "foundationpose",
            "foundationpose_02": "foundationpose",
            "ca1m_01": "ca1m_objects",
        }
        context_sources = {context.context_id: context.source for context in self.contexts}
        if context_sources != expected:
            raise ValueError(f"contexts must exactly match {expected}")
        if len(self.trajectories) != 50:
            raise ValueError("dataset must contain exactly 50 trajectories")
        for context_id in expected:
            subset = [item for item in self.trajectories if item.context_id == context_id]
            if len(subset) != 10:
                raise ValueError(f"{context_id} must contain exactly 10 trajectories")
        ids = [trajectory.trajectory_id for trajectory in self.trajectories]
        if len(ids) != len(set(ids)):
            raise ValueError("trajectory IDs must be unique")
        return self

    def validate_files(self, dataset_root: Path) -> list[Path]:
        missing: list[Path] = []
        for context in self.contexts:
            relative_paths = [
                context.mesh_path,
                context.camera_path,
                context.point_cloud_path,
                *context.rgb_paths,
                *context.depth_paths,
            ]
            for relative in relative_paths:
                path = dataset_root / relative
                if not path.is_file():
                    missing.append(path)
        for trajectory in self.trajectories:
            for turn in trajectory.turns:
                for relative in turn.observation_paths:
                    path = dataset_root / relative
                    if not path.is_file():
                        missing.append(path)
        return missing
