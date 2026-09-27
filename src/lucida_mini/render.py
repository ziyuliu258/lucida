from __future__ import annotations

import atexit
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .metrics import oriented_box_corners
from .pose import Pose
from .state import ObservationMode


BOX_EDGES = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if bin(i ^ j).count("1") == 1
]
TILE_SIZE = 384
ORTHO_TILE_SIZE = 320
MESH_RGB = np.array([255, 126, 18], dtype=np.float32)
HIDDEN_RGB = np.array([35, 220, 85], dtype=np.float32)
POINT_RGB = np.array([115, 193, 216], dtype=np.float32)
AXIS_RGB = ((255, 45, 45), (45, 220, 70), (50, 110, 255))


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

    def scaled(self, width: int, height: int) -> "CameraSpec":
        sx, sy = width / self.width, height / self.height
        if self.projection == "perspective":
            return CameraSpec(
                self.camera_to_world_gl,
                width,
                height,
                self.projection,
                fx=self.fx * sx,
                fy=self.fy * sy,
                cx=self.cx * sx,
                cy=self.cy * sy,
            )
        return CameraSpec(
            self.camera_to_world_gl,
            width,
            height,
            self.projection,
            xmag=self.xmag,
            ymag=self.ymag,
        )

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        camera = (self.camera_to_world_gl[:3, :3].T @ (points - self.camera_to_world_gl[:3, 3]).T).T
        if self.projection == "perspective":
            depth = -camera[:, 2]
            valid = depth > 1e-6
            pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
            pixels[valid, 0] = self.fx * camera[valid, 0] / depth[valid] + self.cx
            pixels[valid, 1] = self.cy - self.fy * camera[valid, 1] / depth[valid]
            return pixels, depth
        depth = -camera[:, 2]
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
def _mesh_for_render(path: str, renderer_width: int, renderer_height: int, use_source_material: bool = False):
    import pyrender
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    if use_source_material:
        # Keep the GLB's UV map and PBR texture for presentation renders.
        return pyrender.Mesh.from_trimesh(mesh, smooth=True)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(1.0, 0.37, 0.035, 1.0),
        metallicFactor=0.0,
        roughnessFactor=0.9,
        alphaMode="OPAQUE",
        doubleSided=True,
    )
    return pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)


@lru_cache(maxsize=16)
def _point_cloud(path: str) -> np.ndarray:
    points = np.asarray(np.load(path), dtype=np.float64).reshape(-1, 3)
    if len(points) > 3500:
        points = points[np.linspace(0, len(points) - 1, 3500, dtype=np.int64)]
    return points


def _pose_matrix(pose: Pose) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = pose.rotation @ np.diag(pose.size)
    transform[:3, 3] = pose.position
    return transform


def _render_depth(
    mesh_path: Path, pose: Pose, spec: CameraSpec, use_source_material: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    import pyrender

    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        ambient_light=np.array([1.0, 1.0, 1.0]),
    )
    scene.add(
        _mesh_for_render(
            str(mesh_path), spec.width, spec.height,
            use_source_material=use_source_material,
        ),
        pose=_pose_matrix(pose),
    )
    camera = _camera_for_pyrender(spec)
    scene.add(camera, pose=spec.camera_to_world_gl)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=2.5),
        pose=spec.camera_to_world_gl,
    )
    color, depth = _renderer(spec.width, spec.height).render(scene)
    depth = np.asarray(depth, dtype=np.float32)
    if use_source_material:
        return depth, np.asarray(color[..., :3], dtype=np.uint8)
    return depth


def _foundation_camera(camera: dict) -> CameraSpec:
    width, height = camera["resolution_wh"]
    view_row = np.asarray(camera["world_to_camera_row_major"], dtype=np.float64)
    projection = np.asarray(camera["projection_row_major"], dtype=np.float64)
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


def _ca1m_camera(camera: dict) -> CameraSpec:
    width, height = camera["resolution_wh"]
    camera_to_world_cv = np.asarray(camera["camera_to_world"], dtype=np.float64)
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    intrinsic = np.asarray(camera["rgb_intrinsic"], dtype=np.float64)
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


def _fit_viewport(width: int, height: int, box: int = TILE_SIZE) -> tuple[int, int]:
    scale = min(box / width, box / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _mesh_overlay(
    background: Image.Image,
    mesh_path: Path,
    pose: Pose,
    spec: CameraSpec,
    observed_depth: np.ndarray | None,
    point_cloud: np.ndarray | None,
    use_source_material: bool = False,
) -> Image.Image:
    width, height = spec.width, spec.height
    base = np.asarray(background.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)).copy()

    if point_cloud is not None and len(point_cloud):
        pixels, depths = spec.project(point_cloud)
        xy = np.rint(pixels).astype(np.int64, casting="unsafe")
        valid = (
            np.isfinite(pixels).all(axis=1)
            & (depths > 0)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )
        cloud_layer = base.copy()
        cloud_layer[xy[valid, 1], xy[valid, 0]] = POINT_RGB.astype(np.uint8)
        base = cv2.addWeighted(base, 0.84, cloud_layer, 0.16, 0)

    spec = spec.scaled(width, height)
    render = _render_depth(mesh_path, pose, spec, use_source_material)
    if use_source_material:
        mesh_depth, textured_color = render
    else:
        mesh_depth, textured_color = render, None
    visible = mesh_depth > 1e-6
    hidden = np.zeros_like(visible)
    if observed_depth is not None:
        sensor = np.asarray(observed_depth, dtype=np.float32)
        if sensor.shape != (height, width):
            sensor = cv2.resize(sensor, (width, height), interpolation=cv2.INTER_NEAREST)
        valid_sensor = np.isfinite(sensor) & (sensor > 0)
        hidden = visible & valid_sensor & (mesh_depth > sensor + 0.015)
    shown = visible & ~hidden
    if textured_color is not None:
        # The display-only terminal image shows the asset's own mapped color
        # and preserves the original scene at pixel locations outside the mesh.
        alpha = 0.88
        base[shown] = (
            base[shown].astype(np.float32) * (1.0 - alpha)
            + textured_color[shown].astype(np.float32) * alpha
        ).astype(np.uint8)
        base[hidden] = (
            textured_color[hidden].astype(np.float32) * 0.48
            + np.asarray(HIDDEN_RGB) * 0.52
        ).astype(np.uint8)
    else:
        alpha = 0.48
        base[shown] = (
            base[shown].astype(np.float32) * (1.0 - alpha) + MESH_RGB * alpha
        ).astype(np.uint8)
        base[hidden] = (
            base[hidden].astype(np.float32) * (1.0 - alpha) + HIDDEN_RGB * alpha
        ).astype(np.uint8)

    overlay = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(overlay)
    _draw_gizmo(draw, spec, pose, show_bounds=not use_source_material)
    return overlay


def _draw_gizmo(
    draw: ImageDraw.ImageDraw, spec: CameraSpec, pose: Pose, show_bounds: bool = True
) -> None:
    if show_bounds:
        corners = oriented_box_corners(pose)
        box_pixels, box_depth = spec.project(corners)
        for first, second in BOX_EDGES:
            if box_depth[first] > 0 or box_depth[second] > 0:
                draw.line(
                    [tuple(box_pixels[first]), tuple(box_pixels[second])],
                    fill=(255, 222, 0),
                    width=max(1, spec.width // 170),
                )

    axis_length = 0.7 * float(np.min(pose.size))
    axes = np.vstack(
        [pose.position, *(pose.position + pose.rotation[:, index] * axis_length for index in range(3))]
    )
    pixels, depths = spec.project(axes)
    for index, color in enumerate(AXIS_RGB, start=1):
        if depths[0] > 0 and depths[index] > 0 and np.isfinite(pixels[[0, index]]).all():
            draw.line(
                [tuple(pixels[0]), tuple(pixels[index])],
                fill=color,
                width=max(2, spec.width // 90),
            )


def _panel(image: Image.Image, label: str, size: int = TILE_SIZE) -> Image.Image:
    panel = Image.new("RGB", (size, size), (238, 240, 243))
    inner = ImageOps.contain(image.convert("RGB"), (size, size - 28), Image.Resampling.LANCZOS)
    x = (size - inner.width) // 2
    y = 24 + (size - 28 - inner.height) // 2
    panel.paste(inner, (x, y))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, size, 22), fill=(24, 28, 34))
    draw.text((8, 4), label, fill=(255, 255, 255))
    return panel


def _font(size: int = 23):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _title(canvas: Image.Image, title: str) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 32), fill=(255, 255, 255))
    draw.text((8, 5), title, fill=(15, 15, 15), font=_font(20))


def _save_panels(output: Path, title: str, panels: list[Image.Image], columns: int) -> None:
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * TILE_SIZE, rows * TILE_SIZE + 34), "white")
    _title(canvas, title)
    for index, panel in enumerate(panels):
        canvas.paste(
            panel,
            ((index % columns) * TILE_SIZE, 34 + (index // columns) * TILE_SIZE),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def _reference_panel(
    image_path: Path,
    bbox: list[float] | None,
    label: str,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if bbox is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(bbox), outline=(255, 222, 0), width=max(2, image.width // 250))
    return _panel(image, label)


def _label_native(image: Image.Image, label: str) -> Image.Image:
    labeled = image.convert("RGB").copy()
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, 22), fill=(24, 28, 34))
    draw.text((8, 4), label, fill=(255, 255, 255))
    return labeled


def _focus_crop_box(
    bbox: list[float], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Use one fixed target-centered crop for every state of a context."""
    width, height = image_size
    target = np.asarray(bbox, dtype=np.float64)
    center = (target[:2] + target[2:]) / 2
    side = int(np.ceil(max(96.0, 4.0 * float(np.max(target[2:] - target[:2])))))
    side = min(side, width, height)
    left = int(np.clip(round(center[0] - side / 2), 0, width - side))
    top = int(np.clip(round(center[1] - side / 2), 0, height - side))
    return left, top, left + side, top + side


def _save_native_scene_set(
    output: Path,
    source: Image.Image,
    current: Image.Image,
    bbox: list[float],
) -> list[Path]:
    """Keep both full frames at native resolution and add a shared target crop."""
    reference = source.convert("RGB").copy()
    draw = ImageDraw.Draw(reference)
    draw.rectangle(tuple(bbox), outline=(255, 222, 0), width=max(2, source.width // 250))
    crop_box = _focus_crop_box(bbox, source.size)
    crop_size = 256
    focus = Image.new("RGB", (2 * crop_size, crop_size), "white")
    for index, frame in enumerate((reference, current)):
        cropped = frame.crop(crop_box).resize(
            (crop_size, crop_size), Image.Resampling.LANCZOS
        )
        focus.paste(cropped, (index * crop_size, 0))
    focus = _label_native(focus, "REFERENCE CROP                    CURRENT MESH CROP")
    paths = [
        output.with_name(f"{output.stem}_reference.png"),
        output.with_name(f"{output.stem}_current.png"),
        output.with_name(f"{output.stem}_focus.png"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, image in zip(
        paths,
        (
            _label_native(reference, "REFERENCE RGB / TARGET BOX"),
            _label_native(current, "CURRENT MESH / SCENE"),
            focus,
        ),
    ):
        image.save(path, optimize=True)
    return paths


def _observation_depth(context_dir: Path, kind: str, view_index: int = 0) -> np.ndarray | None:
    if kind == "foundationpose":
        path = context_dir / "depth_m.npy"
        return np.load(path).astype(np.float32) if path.exists() else None
    if kind == "ca1m_objects":
        path = context_dir / f"view_{view_index:02d}_depth_mm.png"
        if not path.exists():
            return None
        return np.asarray(Image.open(path), dtype=np.float32) / 1000.0
    return None


def _load_cloud(context_dir: Path) -> np.ndarray | None:
    path = context_dir / "point_cloud_world_m.npy"
    return _point_cloud(str(path)) if path.exists() else None


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
    radius = max(float(np.max(pose.size)) * 0.72, 0.01)
    camera_to_world[:3, 3] = pose.position + back_world * max(float(np.max(pose.size)) * 4.0, 1.0)
    return CameraSpec(
        camera_to_world_gl=camera_to_world,
        width=size,
        height=size,
        projection="orthographic",
        xmag=radius,
        ymag=radius,
    )


def _render_six_axis(
    context_dir: Path,
    mesh_path: Path,
    pose: Pose,
    title: str,
) -> Image.Image:
    panels: list[Image.Image] = []
    cloud = _load_cloud(context_dir)
    for axis, axis_name in enumerate(("X", "Y", "Z")):
        for sign, sign_name in ((1, "+"), (-1, "-")):
            spec = _axis_camera(pose, axis, sign, ORTHO_TILE_SIZE)
            depth = _render_depth(mesh_path, pose, spec)
            pixels = Image.new("RGB", (ORTHO_TILE_SIZE, ORTHO_TILE_SIZE), (248, 249, 251))
            array = np.asarray(pixels).copy()
            if cloud is not None and len(cloud):
                cloud_pixels, cloud_depth = spec.project(cloud)
                xy = np.rint(cloud_pixels).astype(np.int64, casting="unsafe")
                valid = (
                    np.isfinite(cloud_pixels).all(axis=1)
                    & (cloud_depth > 0)
                    & (xy[:, 0] >= 0)
                    & (xy[:, 0] < ORTHO_TILE_SIZE)
                    & (xy[:, 1] >= 0)
                    & (xy[:, 1] < ORTHO_TILE_SIZE)
                )
                array[xy[valid, 1], xy[valid, 0]] = POINT_RGB.astype(np.uint8)
            mesh_visible = depth > 1e-6
            array[mesh_visible] = MESH_RGB.astype(np.uint8)
            panel = Image.fromarray(array)
            draw = ImageDraw.Draw(panel)
            _draw_gizmo(draw, spec, pose)
            label = f"LOCAL {sign_name}{axis_name}"
            panel = _panel(panel, label, ORTHO_TILE_SIZE)
            panels.append(panel)

    # Six 320px views occupy 614,400 pixels; the model processor's configured
    # ceiling resizes the contact sheet without multiplying per-turn images.
    rows = 2
    canvas = Image.new(
        "RGB",
        (3 * ORTHO_TILE_SIZE, rows * ORTHO_TILE_SIZE + 34),
        "white",
    )
    _title(canvas, "GizmoAct / SIX LOCAL-AXIS VIEWS")
    for index, panel in enumerate(panels):
        canvas.paste(
            panel,
            ((index // 2) * ORTHO_TILE_SIZE, 34 + (index % 2) * ORTHO_TILE_SIZE),
        )
    return canvas


def _save_observation(
    context_dir: Path,
    pose: Pose,
    output: Path,
    mode: ObservationMode | str,
    title: str,
    kind: str,
    native_focus: bool = False,
    use_source_material: bool = False,
) -> list[Path]:
    mode = ObservationMode(mode)
    context = json.loads((context_dir / "context.json").read_text())
    mesh_path = context_dir / context["mesh_path"]
    if mode is ObservationMode.SIX_AXIS:
        canvas = _render_six_axis(context_dir, mesh_path, pose, title)
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, optimize=True)
        return [output]

    if kind == "foundationpose":
        camera_data = json.loads((context_dir / "camera.json").read_text())
        source = Image.open(context_dir / context["rgb_paths"][0]).convert("RGB")
        width, height = source.size
        view_w, view_h = (
            (width, height) if native_focus else _fit_viewport(width, height)
        )
        spec = _foundation_camera(camera_data).scaled(view_w, view_h)
        current = _mesh_overlay(
            source,
            mesh_path,
            pose,
            spec,
            _observation_depth(context_dir, kind),
            _load_cloud(context_dir),
            use_source_material=use_source_material,
        )
        if native_focus:
            return _save_native_scene_set(
                output, source, current, context["target_bbox_xyxy"]
            )
        panels = [
            _reference_panel(
                context_dir / context["rgb_paths"][0],
                context.get("target_bbox_xyxy"),
                "REFERENCE RGB",
            ),
            _panel(current, "CURRENT MESH / GREEN = OCCLUDED"),
        ]
        _save_panels(output, title, panels, columns=2)
        return [output]

    if kind == "ca1m_objects":
        camera_data = json.loads((context_dir / "camera.json").read_text())["views"]
        panels = []
        cloud = _load_cloud(context_dir)
        for index, view in enumerate(camera_data):
            source_path = context_dir / context["rgb_paths"][index]
            source = Image.open(source_path).convert("RGB")
            view_w, view_h = _fit_viewport(*source.size)
            spec = _ca1m_camera(view).scaled(view_w, view_h)
            current = _mesh_overlay(
                source,
                mesh_path,
                pose,
                spec,
                _observation_depth(context_dir, kind, index),
                cloud,
                use_source_material=use_source_material,
            )
            panels.append(_panel(current, f"CAMERA {index} / CURRENT MESH"))
        _save_panels(output, "GizmoAct / FOUR CALIBRATED RGB-D VIEWS", panels, columns=2)
        return [output]

    if kind == "populated_3d_front":
        background = Image.open(context_dir / context["rgb_paths"][0]).convert("RGB")
        width, height = background.size
        viewport_w, viewport_h = (
            (width, height) if native_focus else _fit_viewport(width, height)
        )
        bbox = np.asarray(context["target_bbox_xyxy"], dtype=np.float64)
        center_px = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        box_wh = np.maximum([bbox[2] - bbox[0], bbox[3] - bbox[1]], 8.0)
        view_x = np.array([0.94, -0.34, 0.0], dtype=np.float64)
        view_y = np.array([0.18, 0.49, -0.85], dtype=np.float64)
        right = view_x / np.linalg.norm(view_x)
        up = view_y - right * float(right @ view_y)
        up /= np.linalg.norm(up)
        back = np.cross(right, up)
        back /= np.linalg.norm(back)
        target = Pose(
            np.asarray(context["target_pose"]["position_m"], dtype=np.float64),
            np.asarray(context["target_pose"]["rotation_object_to_world"], dtype=np.float64),
            np.asarray(context["target_pose"]["size_m"], dtype=np.float64),
        )
        scale = min(box_wh[0] / max(target.size.max(), 1e-6), box_wh[1] / max(target.size.max(), 1e-6)) * 1.25
        scale *= viewport_w / width
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, :3] = np.column_stack((right, up, back))
        camera_to_world[:3, 3] = target.position + back * 10.0
        spec = CameraSpec(
            camera_to_world_gl=camera_to_world,
            width=viewport_w,
            height=viewport_h,
            projection="orthographic",
            xmag=viewport_w / (2.0 * scale),
            ymag=viewport_h / (2.0 * scale),
        )
        # Keep the released rectangle center while using the deterministic
        # oblique projection documented for MesaTask's missing camera data.
        base_center = center_px * (viewport_w / width)
        spec = CameraSpec(
            spec.camera_to_world_gl,
            spec.width,
            spec.height,
            spec.projection,
            xmag=spec.xmag,
            ymag=spec.ymag,
        )
        # Shift the camera so the target projects onto the original target box
        # center rather than the center of the square render viewport.
        desired_center = np.array([base_center[0], base_center[1]])
        actual_center, _ = spec.project(target.position[None, :])
        shift = desired_center - actual_center[0]
        if np.isfinite(shift).all() and np.linalg.norm(shift) > 0.5:
            matrix = spec.camera_to_world_gl.copy()
            matrix[:3, 3] -= matrix[:3, 0] * shift[0] / scale
            matrix[:3, 3] += matrix[:3, 1] * shift[1] / scale
            spec = CameraSpec(
                matrix, spec.width, spec.height, spec.projection,
                xmag=spec.xmag, ymag=spec.ymag,
            )
        current = _mesh_overlay(
            background, mesh_path, pose, spec, None, _load_cloud(context_dir),
            use_source_material=use_source_material,
        )
        if native_focus:
            return _save_native_scene_set(
                output, background, current, context["target_bbox_xyxy"]
            )
        panels = [
            _reference_panel(
                context_dir / context["rgb_paths"][0],
                context.get("target_bbox_xyxy"),
                "REFERENCE RGB",
            ),
            _panel(current, "CURRENT MESH + SCENE POINT CLOUD"),
        ]
        _save_panels(output, "GizmoAct / APPROXIMATE MESA DISPLAY CAMERA", panels, columns=2)
        return [output]

    raise NotImplementedError(f"no renderer for source type {kind!r}")


def render_native_focus_observation(
    context_dir: Path,
    pose: Pose,
    output: Path,
    mode: ObservationMode | str = ObservationMode.SCENE,
    use_source_material: bool = False,
) -> list[Path]:
    """Render a scene observation; optionally preserve the source GLB material for display."""
    context = json.loads((context_dir / "context.json").read_text())
    return _save_observation(
        context_dir,
        pose,
        output,
        mode,
        "Adjust the highlighted target mesh",
        context["source"],
        native_focus=True,
        use_source_material=use_source_material,
    )


def render_foundation_observation(
    context_dir: Path,
    pose: Pose,
    output: Path,
    title: str = "Adjust the highlighted target mesh",
    mode: ObservationMode | str = ObservationMode.SCENE,
) -> None:
    """Render the calibrated FoundationPose scene and the full transformed mesh surface."""
    _save_observation(context_dir, pose, output, mode, title, "foundationpose")


def render_mesatask_observation(
    context_dir: Path,
    pose: Pose,
    output: Path,
    title: str = "Adjust the highlighted target mesh",
    mode: ObservationMode | str = ObservationMode.SCENE,
) -> None:
    """Render MesaTask with its documented fixed approximate display projection."""
    _save_observation(context_dir, pose, output, mode, title, "populated_3d_front")


def render_ca1m_observation(
    context_dir: Path,
    pose: Pose,
    output: Path,
    title: str = "Adjust the highlighted target mesh",
    mode: ObservationMode | str = ObservationMode.SCENE,
) -> None:
    """Render the current mesh surface in all calibrated CA-1M RGB-D views."""
    _save_observation(context_dir, pose, output, mode, title, "ca1m_objects")
