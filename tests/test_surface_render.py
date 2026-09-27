from __future__ import annotations

import json

import numpy as np
import trimesh
from PIL import Image

from lucida_mini.pose import Pose
from lucida_mini.render import (
    CameraSpec,
    _mesh_overlay,
    render_mesatask_observation,
    render_native_focus_observation,
)
from lucida_mini.state import ObservationMode


def make_mesatask_context(root):
    root.mkdir(parents=True)
    trimesh.creation.box(extents=[1.0, 0.45, 0.3]).export(root / "mesh.glb")
    Image.new("RGB", (256, 256), (225, 225, 225)).save(root / "rgb.png")
    np.save(root / "point_cloud_world_m.npy", np.empty((0, 3), dtype=np.float32))
    context = {
        "source": "populated_3d_front",
        "mesh_path": "mesh.glb",
        "rgb_paths": ["rgb.png"],
        "target_bbox_xyxy": [64, 64, 192, 192],
        "target_pose": {
            "position_m": [0.0, 0.0, 0.0],
            "rotation_object_to_world": np.eye(3).tolist(),
            "size_m": [1.0, 0.45, 0.3],
        },
    }
    (root / "context.json").write_text(json.dumps(context))
    return root


def test_surface_observation_changes_when_mesh_moves_and_switches_view(tmp_path):
    context = make_mesatask_context(tmp_path / "context")
    base = Pose.identity(size=np.array([1.0, 0.45, 0.3]))
    moved = Pose(np.array([0.15, 0.0, 0.0]), base.rotation, base.size)
    scene_path = tmp_path / "scene.png"
    moved_path = tmp_path / "moved.png"
    axis_path = tmp_path / "axis.png"

    render_mesatask_observation(context, base, scene_path)
    render_mesatask_observation(context, moved, moved_path)
    render_mesatask_observation(
        context, base, axis_path, mode=ObservationMode.SIX_AXIS
    )

    scene = np.asarray(Image.open(scene_path))
    moved_image = np.asarray(Image.open(moved_path))
    axis = np.asarray(Image.open(axis_path))
    assert scene.shape != axis.shape
    assert np.count_nonzero(np.any(scene != moved_image, axis=2)) > 1000

    # A filled triangle raster should cover a broad surface area, not only the
    # handful of projected mesh vertices used by the old point renderer.
    current_panel = scene[34:, 384:]
    orange_surface = (
        (current_panel[..., 0].astype(int) - current_panel[..., 1].astype(int) > 30)
        & (current_panel[..., 1].astype(int) - current_panel[..., 2].astype(int) > 20)
    )
    assert orange_surface.sum() > 1200

    axis_surface = (
        (axis[..., 0].astype(int) - axis[..., 1].astype(int) > 60)
        & (axis[..., 0].astype(int) - axis[..., 2].astype(int) > 70)
    )
    assert axis_surface.sum() > 2000


def test_depth_buffer_marks_mesh_hidden_by_observed_surface(tmp_path):
    context = make_mesatask_context(tmp_path / "context")
    pose = Pose(np.array([0.0, 0.0, -2.0]), np.eye(3), np.array([0.8, 0.8, 0.8]))
    camera = CameraSpec(
        camera_to_world_gl=np.eye(4),
        width=128,
        height=128,
        projection="perspective",
        fx=110,
        fy=110,
        cx=64,
        cy=64,
    )
    background = Image.new("RGB", (128, 128), (80, 80, 80))
    observed_depth = np.full((128, 128), 1.0, dtype=np.float32)
    rendered = np.asarray(
        _mesh_overlay(
            background,
            context / "mesh.glb",
            pose,
            camera,
            observed_depth,
            None,
        )
    )
    green = (
        (rendered[..., 1].astype(int) > rendered[..., 0].astype(int) + 25)
        & (rendered[..., 1].astype(int) > rendered[..., 2].astype(int) + 25)
    )
    assert green.sum() > 500


def test_native_focus_keeps_full_scene_pixels_and_changes_after_pose_update(tmp_path):
    context = make_mesatask_context(tmp_path / "context")
    Image.new("RGB", (512, 512), (225, 225, 225)).save(context / "rgb.png")
    base = Pose.identity(size=np.array([1.0, 0.45, 0.3]))
    moved = Pose(np.array([0.15, 0.0, 0.0]), base.rotation, base.size)
    base_paths = render_native_focus_observation(context, base, tmp_path / "base.png")
    moved_paths = render_native_focus_observation(context, moved, tmp_path / "moved.png")

    assert [Image.open(path).size for path in base_paths] == [
        (512, 512), (512, 512), (512, 256)
    ]
    assert np.array_equal(
        np.asarray(Image.open(base_paths[0])), np.asarray(Image.open(moved_paths[0]))
    )
    assert np.array_equal(
        np.asarray(Image.open(base_paths[2]))[:, :256],
        np.asarray(Image.open(moved_paths[2]))[:, :256],
    )
    assert np.count_nonzero(
        np.any(
            np.asarray(Image.open(base_paths[1]))
            != np.asarray(Image.open(moved_paths[1])),
            axis=2,
        )
    ) > 1000
