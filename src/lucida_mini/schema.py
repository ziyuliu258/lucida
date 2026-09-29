from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from PIL import Image

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
    step: int = Field(ge=0, le=63)
    observation_paths: list[str] = Field(min_length=1)
    observation_frame_index: int | None = Field(default=None, ge=0)
    observation_mode_before: ObservationMode = ObservationMode.SCENE
    observation_mode_after: ObservationMode = ObservationMode.SCENE
    action: str
    injected_error: bool = False
    injected_error_type: str | None = None
    supervise: bool = True
    state_before: PoseRecord
    state_after: PoseRecord

    @model_validator(mode="after")
    def injected_errors_are_masked(self) -> "TurnRecord":
        if self.injected_error and self.supervise:
            raise ValueError("injected error actions must not be supervised")
        if self.injected_error_type is not None and not self.injected_error:
            raise ValueError("injected_error_type is only valid for injected actions")
        is_switch = self.action.startswith("<switch_obs>")
        if is_switch and self.state_before != self.state_after:
            raise ValueError("switch_obs must leave the object pose unchanged")
        expected_mode = (
            self.observation_mode_before.toggled()
            if is_switch
            else self.observation_mode_before
        )
        if self.observation_mode_after is not expected_mode:
            raise ValueError("observation mode must change only when switch_obs is executed")
        return self


class TrajectoryRecord(BaseModel):
    trajectory_id: str
    context_id: str
    seed: int
    initial_pose: PoseRecord
    target_pose: PoseRecord
    turns: list[TurnRecord] = Field(min_length=1, max_length=64)
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
    point_cloud_paths_by_view: list[str] = Field(default_factory=list, max_length=4)
    scene_identifier: str | None = None
    support_surface_id: str | None = None
    layout_to_world: list[list[float]] | None = None
    scene_geometry_path: str | None = None
    layout_path: str | None = None
    source_split: Literal["train", "validation", "test"] | None = None
    asset_alignment_method: str | None = None
    alignment_provenance: dict | None = None
    alignment_path: str | None = None
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
                *context.point_cloud_paths_by_view,
                *context.rgb_paths,
                *context.depth_paths,
            ]
            if context.scene_geometry_path:
                relative_paths.append(context.scene_geometry_path)
            if context.layout_path:
                relative_paths.append(context.layout_path)
            if context.alignment_path:
                relative_paths.append(context.alignment_path)
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

    def validate_corrected_inputs(self, dataset_root: Path) -> None:
        """Refuse training from the known MesaTask proxy or incomplete CA-1M data."""
        for context in self.contexts:
            camera_path = dataset_root / context.camera_path
            camera = json.loads(camera_path.read_text())
            if context.source == "populated_3d_front":
                required = {
                    "scene_identifier": context.scene_identifier,
                    "support_surface_id": context.support_surface_id,
                    "layout_to_world": context.layout_to_world,
                    "scene_geometry_path": context.scene_geometry_path,
                }
                missing = [name for name, value in required.items() if value is None]
                if missing:
                    raise ValueError(
                        f"{context.context_id} is the old MesaTask proxy; missing real "
                        f"3D-FRONT metadata: {missing}"
                    )
                if camera.get("type") == "display_projection" or not {
                    "camera_to_world", "rgb_intrinsic", "resolution_wh"
                }.issubset(camera):
                    raise ValueError(
                        f"{context.context_id} has no calibrated RGB camera for its combined scene"
                    )
                if not (dataset_root / context.scene_geometry_path).is_file():
                    raise FileNotFoundError(dataset_root / context.scene_geometry_path)
            if context.source == "ca1m_objects":
                if context.source_split != "train":
                    raise ValueError(
                        f"{context.context_id} must record CA-1M train split provenance"
                    )
                if context.asset_alignment_method != "manual_lidar_pointcloud_alignment":
                    raise ValueError(
                        f"{context.context_id} must record a manual SAM 3D to LiDAR point-cloud alignment"
                    )
                provenance = context.alignment_provenance or {}
                if not context.alignment_path or not all(
                    provenance.get(key) for key in ("sha256", "annotator", "method")
                ):
                    raise ValueError(
                        f"{context.context_id} must retain the manual alignment file and its provenance"
                    )
                views = camera.get("views", [])
                if len(views) != len(context.rgb_paths) or len(views) != len(context.depth_paths):
                    raise ValueError(
                        f"{context.context_id} needs one RGB-D calibration record per source frame"
                    )
                if any("target_bbox_xyxy" not in view for view in views):
                    raise ValueError(
                        f"{context.context_id} camera views need frame-specific target boxes"
                    )
                if any("camera_to_world" not in view or "rgb_intrinsic" not in view for view in views):
                    raise ValueError(f"{context.context_id} contains an uncalibrated RGB view")
                for frame_index, (view, relative_rgb) in enumerate(zip(views, context.rgb_paths)):
                    if "resolution_wh" not in view:
                        raise ValueError(
                            f"{context.context_id} view {frame_index} has no recorded native resolution"
                        )
                    image_path = dataset_root / relative_rgb
                    with Image.open(image_path) as image:
                        actual_resolution = [image.width, image.height]
                    recorded_resolution = [int(value) for value in view["resolution_wh"]]
                    if actual_resolution != recorded_resolution:
                        raise ValueError(
                            f"{context.context_id} view {frame_index} RGB resolution changed: "
                            f"recorded {recorded_resolution}, actual {actual_resolution}"
                        )
                    if any(value % 32 for value in recorded_resolution):
                        raise ValueError(
                            f"{context.context_id} view {frame_index} native resolution must "
                            "be divisible by 32 to prevent Qwen resizing"
                        )
                alignment_file = dataset_root / context.alignment_path
                alignment_bytes = alignment_file.read_bytes()
                if hashlib.sha256(alignment_bytes).hexdigest() != provenance["sha256"]:
                    raise ValueError(
                        f"{context.context_id} manual alignment file hash does not match its manifest"
                    )
                alignment = json.loads(alignment_bytes)
                if (
                    alignment.get("annotator") != provenance["annotator"]
                    or alignment.get("method") != provenance["method"]
                ):
                    raise ValueError(
                        f"{context.context_id} manual alignment metadata does not match its manifest"
                    )
                recorded_pose = PoseRecord.model_validate(alignment.get("target_pose", {}))
                if recorded_pose != context.target_pose:
                    raise ValueError(
                        f"{context.context_id} target pose differs from its manual LiDAR alignment"
                    )

        # Every new turn uses three separate main views. SIX_AXIS adds six
        # separately saved orthographic views; old contact sheets do not meet
        # this input contract. CA-1M selects a single source frame per turn.
        for trajectory in self.trajectories:
            context = next(item for item in self.contexts if item.context_id == trajectory.context_id)
            frame_count = len(context.rgb_paths)
            for turn in trajectory.turns:
                expected_count = 3 + (6 if turn.observation_mode_before is ObservationMode.SIX_AXIS else 0)
                if len(turn.observation_paths) != expected_count:
                    raise ValueError(
                        f"{trajectory.trajectory_id} step {turn.step} must contain "
                        f"{expected_count} separate observation views, found {len(turn.observation_paths)}"
                    )
                raw_path = Path(turn.observation_paths[0])
                expected_paths = [
                    raw_path.as_posix(),
                    raw_path.with_name(f"{raw_path.stem}_overlay{raw_path.suffix}").as_posix(),
                    raw_path.with_name(f"{raw_path.stem}_pointcloud{raw_path.suffix}").as_posix(),
                ]
                if turn.observation_mode_before is ObservationMode.SIX_AXIS:
                    expected_paths.extend(
                        raw_path.with_name(
                            f"{raw_path.stem}_local_{axis}_{sign}{raw_path.suffix}"
                        ).as_posix()
                        for axis in ("x", "y", "z")
                        for sign in ("pos", "neg")
                    )
                if turn.observation_paths != expected_paths:
                    raise ValueError(
                        f"{trajectory.trajectory_id} step {turn.step} observation paths "
                        "must be separate raw, overlay, point-cloud and local-axis files"
                    )
                expected_frame = min(turn.step, frame_count - 1)
                if turn.observation_frame_index != expected_frame:
                    raise ValueError(
                        f"{trajectory.trajectory_id} step {turn.step} must select source frame "
                        f"{expected_frame}, found {turn.observation_frame_index}"
                    )
