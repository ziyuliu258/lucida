from __future__ import annotations

import atexit
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw

from .metrics import oriented_box_corners
from .pose import Pose
from .state import ObservationMode


BOX_EDGES = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if bin(i ^ j).count("1") == 1
]
ORTHO_SIZE = 320
HIDDEN_RGB = np.array([54, 224, 128], dtype=np.uint8)
AXIS_RGB = ((255, 45, 45), (45, 220, 70), (50, 110, 255))
POINT_BACKGROUND = (246, 247, 249)


@dataclass(frozen=True)
class CameraSpec:
    camera_to_world_gl: np.ndarray
    width: int
    height: int
    projection: str
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    xmag: float = 0.0
    ymag: float = 0.0

    def __post_init__(self) -> None:
        transform = np.asarray(self.camera_to_world_gl, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("camera_to_world_gl must be a finite 4x4 matrix")
        rotation = transform[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
            or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        ):
            raise ValueError("camera_to_world_gl must be a rigid right-handed transform")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera resolution must be positive")
        if self.projection == "perspective" and (
            self.fx <= 0 or self.fy <= 0 or not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all()
        ):
            raise ValueError("perspective camera requires finite positive focal lengths")
        if self.projection == "orthographic" and (
            self.xmag <= 0 or self.ymag <= 0
        ):
            raise ValueError("orthographic camera requires positive extents")

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        camera = (
            self.camera_to_world_gl[:3, :3].T
            @ (points - self.camera_to_world_gl[:3, 3]).T
        ).T
        depth = -camera[:, 2]
        if self.projection == "perspective":
            valid = depth > 1e-6
            pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
            pixels[valid, 0] = self.fx * camera[valid, 0] / depth[valid] + self.cx
            pixels[valid, 1] = self.cy - self.fy * camera[valid, 1] / depth[valid]
            return pixels, depth
        pixels = np.column_stack(
            (
                self.width / 2 + camera[:, 0] * self.width / (2 * self.xmag),
                self.height / 2 - camera[:, 1] * self.height / (2 * self.ymag),
            )
        )
        return pixels, depth


def _camera_for_pyrender(spec: CameraSpec):
    import pyrender

    if spec.projection == "perspective":
        return pyrender.IntrinsicsCamera(
            fx=spec.fx,
            fy=spec.fy,
            cx=spec.cx,
            cy=spec.cy,
            znear=0.001,
            zfar=10000.0,
        )
    return pyrender.OrthographicCamera(
        xmag=spec.xmag,
        ymag=spec.ymag,
        znear=0.001,
        zfar=10000.0,
    )


_RENDERERS: dict[tuple[int, int], object] = {}


def _renderer(width: int, height: int):
    import pyrender

    key = (width, height)
    if key not in _RENDERERS:
        _RENDERERS[key] = pyrender.OffscreenRenderer(width, height)
    return _RENDERERS[key]


def _delete_renderers() -> None:
    for renderer in _RENDERERS.values():
        try:
            renderer.delete()
        except Exception:
            pass


atexit.register(_delete_renderers)


@lru_cache(maxsize=32)
def _meshes_for_render(path: str) -> tuple[tuple[object, np.ndarray], ...]:
    import pyrender
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        instances = []
        for node in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph.get(node)
            geometry = loaded.geometry[geometry_name]
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces):
                instances.append((geometry, np.asarray(transform, dtype=np.float64)))
    elif isinstance(loaded, trimesh.Trimesh) and len(loaded.faces):
        instances = [(loaded, np.eye(4, dtype=np.float64))]
    else:
        instances = []
    if not instances:
        raise ValueError(f"asset has no triangle geometry: {path}")

    # Keep each scene mesh and its object-local transform separate. Flattening
    # a textured GLB with force="mesh" can lose material and instance data.
    return tuple(
        (pyrender.Mesh.from_trimesh(geometry.copy(), smooth=True), transform)
        for geometry, transform in instances
    )


@lru_cache(maxsize=16)
def _point_cloud(path: str) -> np.ndarray:
    points = np.asarray(np.load(path), dtype=np.float64).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def _pose_matrix(pose: Pose) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = pose.rotation @ np.diag(pose.size)
    transform[:3, 3] = pose.position
    return transform


def _render_mesh(mesh_path: Path, pose: Pose, spec: CameraSpec) -> tuple[np.ndarray, np.ndarray]:
    import pyrender

    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        ambient_light=np.array([1.0, 1.0, 1.0]),
    )
    pose_matrix = _pose_matrix(pose)
    for mesh, asset_transform in _meshes_for_render(str(mesh_path)):
        scene.add(mesh, pose=pose_matrix @ asset_transform)
    scene.add(_camera_for_pyrender(spec), pose=spec.camera_to_world_gl)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=2.5),
        pose=spec.camera_to_world_gl,
    )
    color, depth = _renderer(spec.width, spec.height).render(scene)
    return np.asarray(color[..., :3], dtype=np.uint8), np.asarray(depth, dtype=np.float32)


def _foundation_camera(camera: dict) -> CameraSpec:
    width, height = camera["resolution_wh"]
    view_row = np.asarray(camera["world_to_camera_row_major"], dtype=np.float64)
    projection = np.asarray(camera["projection_row_major"], dtype=np.float64)
    if view_row.shape != (4, 4) or projection.shape != (4, 4):
        raise ValueError("FoundationPose camera matrices must both be 4x4")
    if not (np.isfinite(view_row).all() and np.isfinite(projection).all()):
        raise ValueError("FoundationPose camera matrices must be finite")
    camera_to_world = np.linalg.inv(view_row.T)
    return CameraSpec(
        camera_to_world_gl=camera_to_world,
        width=int(width),
        height=int(height),
        projection="perspective",
        fx=float(projection[0, 0] * width / 2),
        fy=float(projection[1, 1] * height / 2),
        cx=float(width / 2),
        cy=float(height / 2),
    )


def _pinhole_camera(camera: dict) -> CameraSpec:
    if "camera_to_world" not in camera or "rgb_intrinsic" not in camera:
        raise ValueError(
            "scene camera must provide calibrated camera_to_world and rgb_intrinsic; "
            "approximate display projections are not valid training inputs"
        )
    width, height = camera["resolution_wh"]
    camera_to_world_cv = np.asarray(camera["camera_to_world"], dtype=np.float64)
    if camera_to_world_cv.shape != (4, 4):
        raise ValueError("camera_to_world must be a 4x4 calibrated matrix")
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    intrinsic = np.asarray(camera["rgb_intrinsic"], dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError("rgb_intrinsic must be a 3x3 calibrated matrix")
    if not np.isfinite(camera_to_world_cv).all() or not np.isfinite(intrinsic).all():
        raise ValueError("camera calibration values must be finite")
    if (
        not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=1e-6)
        or not np.isclose(intrinsic[0, 1], 0.0, atol=1e-6)
        or not np.isclose(intrinsic[1, 0], 0.0, atol=1e-6)
    ):
        raise ValueError("camera intrinsic must use the supported pinhole matrix form")
    if (
        not np.allclose(camera_to_world_cv[:3, :3].T @ camera_to_world_cv[:3, :3], np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(camera_to_world_cv[:3, :3]), 1.0, atol=1e-4)
        or not np.allclose(camera_to_world_cv[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    ):
        raise ValueError("camera_to_world must be a rigid right-handed transform")
    return CameraSpec(
        camera_to_world_gl=camera_to_world_cv @ cv_to_gl,
        width=int(width),
        height=int(height),
        projection="perspective",
        fx=float(intrinsic[0, 0]),
        fy=float(intrinsic[1, 1]),
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
    )


def _camera_for_source(kind: str, camera: dict, view_index: int) -> CameraSpec:
    if kind == "foundationpose":
        return _foundation_camera(camera)
    if kind == "ca1m_objects":
        views = camera.get("views", [])
        if not views:
            raise ValueError("CA-1M camera.json contains no calibrated views")
        return _pinhole_camera(views[min(view_index, len(views) - 1)])
    if kind == "populated_3d_front":
        return _pinhole_camera(camera)
    raise ValueError(f"unsupported scene source: {kind}")


def _derive_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}_{suffix}{output.suffix}")


def _save_rgb(path: Path, rgb: np.ndarray | Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb.astype(np.uint8), "RGB")
    image.convert("RGB").save(path, optimize=True)


def _draw_gizmo(image: np.ndarray, spec: CameraSpec, pose: Pose) -> np.ndarray:
    canvas = Image.fromarray(image.astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(canvas)
    corners = oriented_box_corners(pose)
    pixels, depths = spec.project(corners)
    line_width = max(2, spec.width // 260)
    for first, second in BOX_EDGES:
        if depths[first] > 0 or depths[second] > 0:
            if np.isfinite(pixels[[first, second]]).all():
                draw.line(
                    [tuple(pixels[first]), tuple(pixels[second])],
                    fill=(255, 220, 0),
                    width=line_width,
                )

    axis_length = 0.7 * float(np.min(pose.size))
    axes = np.vstack(
        [pose.position, *(pose.position + pose.rotation[:, axis] * axis_length for axis in range(3))]
    )
    axis_pixels, axis_depths = spec.project(axes)
    for index, color in enumerate(AXIS_RGB, start=1):
        if axis_depths[0] > 0 and axis_depths[index] > 0 and np.isfinite(axis_pixels[[0, index]]).all():
            draw.line(
                [tuple(axis_pixels[0]), tuple(axis_pixels[index])],
                fill=color,
                width=max(3, spec.width // 120),
            )
    return np.asarray(canvas)


def _project_frontmost_cloud(
    points: np.ndarray,
    spec: CameraSpec,
    source_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project XYZ into the calibrated RGB view and transfer exact pixel colors."""
    if source_rgb.shape[:2] != (spec.height, spec.width):
        raise ValueError(
            f"RGB size {source_rgb.shape[1]}x{source_rgb.shape[0]} does not match "
            f"calibration {spec.width}x{spec.height}"
        )
    pixels, depths = spec.project(points)
    safe_pixels = np.where(np.isfinite(pixels), pixels, -1.0)
    xy = np.rint(safe_pixels).astype(np.int64)
    valid = (
        np.isfinite(pixels).all(axis=1)
        & (depths > 1e-6)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < spec.width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < spec.height)
    )
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    flat = xy[candidates, 1] * spec.width + xy[candidates, 0]
    depth_buffer = np.full(spec.width * spec.height, np.inf, dtype=np.float32)
    np.minimum.at(depth_buffer, flat, depths[candidates].astype(np.float32))
    front = depths[candidates] <= depth_buffer[flat] + 1e-5
    selected = candidates[front]
    selected_xy = xy[selected]
    colors = source_rgb[selected_xy[:, 1], selected_xy[:, 0]].astype(np.uint8)
    return points[selected], colors


def _rasterize_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    spec: CameraSpec,
    background: tuple[int, int, int] = POINT_BACKGROUND,
) -> tuple[np.ndarray, np.ndarray]:
    canvas = np.full((spec.height, spec.width, 3), background, dtype=np.uint8)
    depth_buffer = np.full((spec.height, spec.width), np.inf, dtype=np.float32)
    if not len(points):
        return canvas, depth_buffer
    pixels, depths = spec.project(points)
    safe_pixels = np.where(np.isfinite(pixels), pixels, -1.0)
    xy = np.rint(safe_pixels).astype(np.int64)
    valid = (
        np.isfinite(pixels).all(axis=1)
        & (depths > 1e-6)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < spec.width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < spec.height)
    )
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        return canvas, depth_buffer
    flat = xy[candidates, 1] * spec.width + xy[candidates, 0]
    center_depth = np.full(spec.width * spec.height, np.inf, dtype=np.float32)
    np.minimum.at(center_depth, flat, depths[candidates].astype(np.float32))
    front = depths[candidates] <= center_depth[flat] + 1e-5
    selected = candidates[front]
    x, y = xy[selected, 0], xy[selected, 1]
    rgb = colors[selected]
    z = depths[selected].astype(np.float32)
    # A 3x3 point footprint makes the cloud legible. Build its depth buffer
    # from the same footprint so the green hidden-surface cue has no 1-pixel
    # holes around otherwise visible scene points.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            px, py = x + dx, y + dy
            inside = (px >= 0) & (px < spec.width) & (py >= 0) & (py < spec.height)
            selected_inside = np.flatnonzero(inside)
            if len(selected_inside):
                np.minimum.at(
                    depth_buffer,
                    (py[selected_inside], px[selected_inside]),
                    z[selected_inside],
                )
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            px, py = x + dx, y + dy
            inside = (px >= 0) & (px < spec.width) & (py >= 0) & (py < spec.height)
            selected_inside = np.flatnonzero(inside)
            if len(selected_inside):
                px_in, py_in = px[selected_inside], py[selected_inside]
                visible = z[selected_inside] <= depth_buffer[py_in, px_in] + 1e-5
                points_to_draw = selected_inside[visible]
                canvas[py[points_to_draw], px[points_to_draw]] = rgb[points_to_draw]
    return canvas, depth_buffer


def _blend_mesh(
    background: np.ndarray,
    mesh_rgb: np.ndarray,
    mesh_depth: np.ndarray,
    cloud_depth: np.ndarray | None = None,
    show_occlusion: bool = False,
) -> np.ndarray:
    result = background.copy()
    mesh = mesh_depth > 1e-6
    hidden = np.zeros_like(mesh)
    if show_occlusion and cloud_depth is not None:
        hidden = mesh & np.isfinite(cloud_depth) & (cloud_depth + 0.012 < mesh_depth)
    visible = mesh & ~hidden
    alpha = 0.82
    result[visible] = (
        result[visible].astype(np.float32) * (1.0 - alpha)
        + mesh_rgb[visible].astype(np.float32) * alpha
    ).astype(np.uint8)
    if np.any(hidden):
        tinted = mesh_rgb[hidden].astype(np.float32) * 0.45 + HIDDEN_RGB.astype(np.float32) * 0.55
        result[hidden] = (
            result[hidden].astype(np.float32) * 0.32 + tinted * 0.68
        ).astype(np.uint8)
    return result


def _ca1m_cloud(context_dir: Path, context: dict, view_index: int) -> np.ndarray:
    view = context["views"][view_index]
    cloud_path = view.get("point_cloud_path")
    if cloud_path:
        return _point_cloud(str(context_dir / cloud_path))
    depth_path = context_dir / view["depth_path"]
    depth_mm = np.asarray(Image.open(depth_path), dtype=np.float64)
    intrinsic = np.asarray(view["depth_intrinsic"], dtype=np.float64)
    depth = depth_mm / 1000.0
    rows, cols = np.indices(depth.shape)
    valid = (depth > 0) & np.isfinite(depth) & (depth < 10.0)
    stride = int(context.get("point_cloud_stride", 2))
    valid = valid[::stride, ::stride]
    rows = rows[::stride, ::stride][valid]
    cols = cols[::stride, ::stride][valid]
    z = depth[::stride, ::stride][valid]
    camera_xyz = np.column_stack(
        (
            (cols - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
            np.ones(len(z)),
        )
    )
    camera_to_world = np.asarray(view["camera_to_world"], dtype=np.float64)
    world = (camera_to_world @ camera_xyz.T).T[:, :3]
    return world.astype(np.float64)


def _load_scene_cloud(
    context_dir: Path,
    context: dict,
    kind: str,
    view_index: int,
) -> np.ndarray:
    if kind == "ca1m_objects":
        # Each interaction uses one frame's depth, never a cloud merged from
        # other timestamps in the same capture.
        return _ca1m_cloud(context_dir, context, view_index)
    path = context_dir / context["point_cloud_path"]
    if not path.is_file():
        raise FileNotFoundError(f"scene point cloud is required: {path}")
    return _point_cloud(str(path))


def _axis_camera(pose: Pose, axis: int, sign: int, size: int) -> CameraSpec:
    back_local = np.zeros(3, dtype=np.float64)
    back_local[axis] = float(sign)
    up_local = np.array([0.0, 0.0, 1.0]) if axis != 2 else np.array([0.0, 1.0, 0.0])
    right_local = np.cross(up_local, back_local)
    right_local /= np.linalg.norm(right_local)
    up_local = np.cross(back_local, right_local)
    rotation_local = np.column_stack((right_local, up_local, back_local))
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = pose.rotation @ rotation_local
    back_world = camera_to_world[:3, 2]
    radius = max(float(np.max(pose.size)) * 1.15, 0.01)
    camera_to_world[:3, 3] = pose.position + back_world * (radius * 4.0 + 0.02)
    return CameraSpec(
        camera_to_world_gl=camera_to_world,
        width=size,
        height=size,
        projection="orthographic",
        xmag=radius,
        ymag=radius,
    )


def _validate_populated_scene(context_dir: Path, context: dict, camera: dict) -> None:
    required = ("scene_identifier", "support_surface_id", "layout_to_world", "scene_geometry_path")
    missing = [key for key in required if not context.get(key)]
    if missing:
        raise ValueError(
            "refusing to render the legacy MesaTask proxy as populated_3d_front; "
            f"real MesaTask + 3D-FRONT scene metadata is missing {missing}"
        )
    scene_path = context_dir / context["scene_geometry_path"]
    if not scene_path.is_file():
        raise FileNotFoundError(f"combined 3D-FRONT scene geometry is missing: {scene_path}")
    _pinhole_camera(camera)


def _selected_frame(
    context_dir: Path,
    context: dict,
    camera_data: dict,
    kind: str,
    frame_index: int,
) -> tuple[Image.Image, CameraSpec, np.ndarray]:
    if kind == "ca1m_objects":
        views = camera_data.get("views", [])
        records = context.get("views", [])
        if not views or not records or len(views) != len(records):
            raise ValueError("CA-1M context must contain matching per-frame camera and RGB-D records")
        index = min(max(int(frame_index), 0), len(views) - 1)
        view = records[index]
        source = Image.open(context_dir / view["rgb_path"]).convert("RGB")
        spec = _camera_for_source(kind, camera_data, index)
        cloud = _load_scene_cloud(context_dir, context, kind, index)
    else:
        index = 0
        source = Image.open(context_dir / context["rgb_paths"][0]).convert("RGB")
        spec = _camera_for_source(kind, camera_data, index)
        cloud = _load_scene_cloud(context_dir, context, kind, index)
    if source.size != (spec.width, spec.height):
        raise ValueError(
            f"source RGB is {source.width}x{source.height}, but calibrated camera is "
            f"{spec.width}x{spec.height}"
        )
    return source, spec, cloud


def _render_observation_views(
    context_dir: Path,
    pose: Pose,
    output: Path,
    mode: ObservationMode | str,
    frame_index: int,
) -> list[Path]:
    mode = ObservationMode(mode)
    context = json.loads((context_dir / "context.json").read_text())
    kind = context["source"]
    camera_data = json.loads((context_dir / context.get("camera_path", "camera.json")).read_text())
    if kind == "populated_3d_front":
        _validate_populated_scene(context_dir, context, camera_data)
    mesh_path = context_dir / context["mesh_path"]
    source, camera, cloud = _selected_frame(
        context_dir, context, camera_data, kind, frame_index
    )
    source_rgb = np.asarray(source, dtype=np.uint8)
    colored_points, point_colors = _project_frontmost_cloud(cloud, camera, source_rgb)

    mesh_rgb, mesh_depth = _render_mesh(mesh_path, pose, camera)
    overlay = _blend_mesh(source_rgb, mesh_rgb, mesh_depth)
    overlay = _draw_gizmo(overlay, camera, pose)
    point_view, cloud_depth = _rasterize_cloud(
        colored_points, point_colors, camera, POINT_BACKGROUND
    )
    point_view = _blend_mesh(
        point_view,
        mesh_rgb,
        mesh_depth,
        cloud_depth=cloud_depth,
        show_occlusion=True,
    )
    point_view = _draw_gizmo(point_view, camera, pose)

    paths = [output, _derive_path(output, "overlay"), _derive_path(output, "pointcloud")]
    _save_rgb(paths[0], source_rgb)
    _save_rgb(paths[1], overlay)
    _save_rgb(paths[2], point_view)

    if mode is ObservationMode.SIX_AXIS:
        for axis, axis_name in enumerate(("x", "y", "z")):
            for sign, sign_name in ((1, "pos"), (-1, "neg")):
                ortho = _axis_camera(pose, axis, sign, ORTHO_SIZE)
                ortho_view, ortho_cloud_depth = _rasterize_cloud(
                    colored_points, point_colors, ortho, POINT_BACKGROUND
                )
                ortho_mesh_rgb, ortho_mesh_depth = _render_mesh(mesh_path, pose, ortho)
                ortho_view = _blend_mesh(
                    ortho_view,
                    ortho_mesh_rgb,
                    ortho_mesh_depth,
                    cloud_depth=ortho_cloud_depth,
                    show_occlusion=True,
                )
                ortho_view = _draw_gizmo(ortho_view, ortho, pose)
                path = _derive_path(output, f"local_{axis_name}_{sign_name}")
                _save_rgb(path, ortho_view)
                paths.append(path)
    return paths


def render_native_focus_observation(
    context_dir: Path,
    pose: Pose,
    output: Path,
    mode: ObservationMode | str = ObservationMode.SCENE,
    use_source_material: bool = True,
    frame_index: int = 0,
) -> list[Path]:
    """Write full-resolution raw, textured overlay and RGB-colored point-cloud views."""
    del use_source_material  # All corrected overlays preserve the source asset material.
    return _render_observation_views(context_dir, pose, output, mode, frame_index)


def render_foundation_observation(
    context_dir: Path,
    pose: Pose,
    output: Path,
    title: str = "Adjust the highlighted target mesh",
    mode: ObservationMode | str = ObservationMode.SCENE,
    frame_index: int = 0,
) -> list[Path]:
    del title
    return _render_observation_views(context_dir, pose, output, mode, frame_index)


def render_mesatask_observation(
    context_dir: Path,
    pose: Pose,
    output: Path,
    title: str = "Adjust the highlighted target mesh",
    mode: ObservationMode | str = ObservationMode.SCENE,
    frame_index: int = 0,
) -> list[Path]:
    del title
    return _render_observation_views(context_dir, pose, output, mode, frame_index)


def render_ca1m_observation(
    context_dir: Path,
    pose: Pose,
    output: Path,
    title: str = "Adjust the highlighted target mesh",
    mode: ObservationMode | str = ObservationMode.SCENE,
    frame_index: int = 0,
) -> list[Path]:
    del title
    return _render_observation_views(context_dir, pose, output, mode, frame_index)
