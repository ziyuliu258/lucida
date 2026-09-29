#!/usr/bin/env python3
"""Place MesaTask layouts into calibrated 3D-FRONT scenes and export contexts.

The required scene manifest supplies a real 3D-FRONT GLB, the selected support
surface, the rigid transform from MesaTask's tabletop frame into the scene, and
a calibrated RGB camera.  The old MesaTask-only screenshot and box-sampled
point-cloud fallback has deliberately been removed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


CONTEXTS = {
    "front_01": {
        "layout": "kitchen_counter/kitchen_counter_0813",
        "uid": "5f5fa09d1f6349f993f100abd77d8f80",
        "description": "antique hand-crank coffee grinder",
    },
    "front_02": {
        "layout": "office_table/office_table_0907",
        "uid": "e3774ca0bdae459082f3bed37fa7260c",
        "description": "green clip pen",
    },
}


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def canonical_mesh(path: Path) -> tuple[trimesh.Scene, np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        instances = []
        world_vertices = []
        for node in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph.get(node)
            geometry = loaded.geometry[geometry_name].copy()
            transform = np.asarray(transform, dtype=np.float64)
            instances.append((str(node), geometry, transform))
            homogeneous = np.column_stack((geometry.vertices, np.ones(len(geometry.vertices))))
            world_vertices.append((transform @ homogeneous.T).T[:, :3])
        if not instances:
            raise ValueError(f"asset has no triangle geometry: {path}")
    else:
        instances = [("asset", loaded.copy(), np.eye(4, dtype=np.float64))]
        world_vertices = [np.asarray(loaded.vertices, dtype=np.float64)]
    vertices = np.concatenate(world_vertices, axis=0)
    lower, upper = vertices.min(0), vertices.max(0)
    center, extents = (lower + upper) / 2, upper - lower
    if not np.isfinite(extents).all() or np.any(extents <= 1e-9):
        raise ValueError(f"degenerate object mesh: {path}")
    normalize = np.eye(4, dtype=np.float64)
    normalize[:3, :3] = np.diag(1.0 / extents)
    normalize[:3, 3] = -center / extents
    canonical = trimesh.Scene()
    for index, (node, geometry, transform) in enumerate(instances):
        canonical.add_geometry(
            geometry,
            geom_name=f"asset_geometry_{index:03d}",
            node_name=f"asset_{node}_{index:03d}",
            transform=normalize @ transform,
        )
    return canonical, center, extents


def _layout_transform(record: dict) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(record["layout_to_world"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("layout_to_world must be a finite 4x4 matrix")
    linear = transform[:3, :3]
    if not np.allclose(linear.T @ linear, np.eye(3), atol=1e-5):
        raise ValueError("layout_to_world must be rigid; scale object dimensions in meters")
    if not np.isclose(np.linalg.det(linear), 1.0, atol=1e-5):
        raise ValueError("layout_to_world must be right-handed")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("layout_to_world must be an affine rigid transform")
    return transform, linear


def _mesh_pose(item: dict, layout_to_world: np.ndarray, frame_rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position_layout = np.asarray(item["position"], dtype=np.float64) / 100.0
    size_layout = np.asarray(item["size"], dtype=np.float64) / 100.0
    rotation_layout = Rotation.from_quat(np.asarray(item["rotation"], dtype=np.float64)).as_matrix()
    position_world = (layout_to_world @ np.append(position_layout, 1.0))[:3]
    rotation_world = frame_rotation @ rotation_layout
    return position_world, rotation_world, size_layout


def _pose_matrix(position: np.ndarray, rotation: np.ndarray, size: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(size)
    matrix[:3, 3] = position
    return matrix


def _copy_front_scene(source: trimesh.Scene, output: trimesh.Scene) -> list[tuple[trimesh.Trimesh, np.ndarray]]:
    meshes = []
    for index, node in enumerate(source.graph.nodes_geometry):
        transform, geometry_name = source.graph.get(node)
        geometry = source.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        matrix = np.asarray(transform, dtype=np.float64)
        output.add_geometry(
            geometry.copy(),
            geom_name=f"3dfront_geometry_{index}",
            node_name=f"3dfront_node_{index}",
            transform=matrix,
        )
        meshes.append((geometry, matrix))
    return meshes


def _add_object_scene(
    output: trimesh.Scene,
    asset: trimesh.Scene,
    object_transform: np.ndarray,
    name_prefix: str,
) -> list[tuple[trimesh.Trimesh, np.ndarray]]:
    instances = []
    for index, node in enumerate(asset.graph.nodes_geometry):
        local_transform, geometry_name = asset.graph.get(node)
        geometry = asset.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        world_transform = object_transform @ np.asarray(local_transform, dtype=np.float64)
        output.add_geometry(
            geometry.copy(),
            geom_name=f"{name_prefix}_geometry_{index:03d}",
            node_name=f"{name_prefix}_node_{index:03d}",
            transform=world_transform,
        )
        instances.append((geometry, world_transform))
    if not instances:
        raise ValueError(f"asset {name_prefix} contains no triangle geometry")
    return instances


def _pyrender_scene(mesh_scene: trimesh.Scene, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    width, height = map(int, camera["resolution_wh"])
    intrinsic = np.asarray(camera["rgb_intrinsic"], dtype=np.float64)
    camera_to_world_cv = np.asarray(camera["camera_to_world"], dtype=np.float64)
    if intrinsic.shape != (3, 3) or camera_to_world_cv.shape != (4, 4):
        raise ValueError("camera requires 3x3 rgb_intrinsic and 4x4 camera_to_world")
    if not np.isfinite(intrinsic).all() or not np.isfinite(camera_to_world_cv).all():
        raise ValueError("camera calibration values must be finite")
    if (
        not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=1e-6)
        or not np.isclose(intrinsic[0, 1], 0.0, atol=1e-6)
        or not np.isclose(intrinsic[1, 0], 0.0, atol=1e-6)
    ):
        raise ValueError("camera intrinsic must use the supported pinhole matrix form")
    rotation = camera_to_world_cv[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
        or not np.allclose(camera_to_world_cv[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
    ):
        raise ValueError("camera calibration must contain positive focal lengths and a rigid pose")
    scene = pyrender.Scene(bg_color=np.array([0.96, 0.96, 0.97, 1.0]), ambient_light=np.ones(3) * 0.55)
    for node in mesh_scene.graph.nodes_geometry:
        transform, geometry_name = mesh_scene.graph.get(node)
        geometry = mesh_scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        scene.add(
            pyrender.Mesh.from_trimesh(geometry, smooth=True),
            pose=np.asarray(transform, dtype=np.float64),
        )
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    camera_to_world_gl = camera_to_world_cv @ cv_to_gl
    scene.add(
        pyrender.IntrinsicsCamera(
            fx=float(intrinsic[0, 0]),
            fy=float(intrinsic[1, 1]),
            cx=float(intrinsic[0, 2]),
            cy=float(intrinsic[1, 2]),
            znear=0.001,
            zfar=10000.0,
        ),
        pose=camera_to_world_gl,
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=2.5),
        pose=camera_to_world_gl,
    )
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, depth = renderer.render(scene)
    finally:
        renderer.delete()
    return np.asarray(color[..., :3], dtype=np.uint8), np.asarray(depth, dtype=np.float32)


def _sample_scene_cloud(meshes: list[tuple[trimesh.Trimesh, np.ndarray]], count: int, seed: int) -> np.ndarray:
    world_meshes = []
    for geometry, transform in meshes:
        world = geometry.copy()
        world.apply_transform(transform)
        world_meshes.append(world)
    if not world_meshes:
        raise ValueError("combined scene contains no triangle geometry")
    combined = trimesh.util.concatenate(world_meshes)
    points, _ = trimesh.sample.sample_surface(combined, count, seed=seed)
    return np.asarray(points, dtype=np.float32)


def _project_target_box(pose: dict, camera: dict) -> list[int]:
    position = np.asarray(pose["position_m"], dtype=np.float64)
    rotation = np.asarray(pose["rotation_object_to_world"], dtype=np.float64)
    size = np.asarray(pose["size_m"], dtype=np.float64)
    signs = np.array(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
        dtype=np.float64,
    )
    corners = (rotation @ (signs * size).T).T + position
    camera_to_world = np.asarray(camera["camera_to_world"], dtype=np.float64)
    camera_points = (np.linalg.inv(camera_to_world) @ np.column_stack((corners, np.ones(8))).T).T[:, :3]
    in_front = camera_points[:, 2] > 1e-6
    if not in_front.any():
        raise ValueError("MesaTask target is behind the selected RGB camera")
    intrinsic = np.asarray(camera["rgb_intrinsic"], dtype=np.float64)
    visible = camera_points[in_front]
    x = intrinsic[0, 0] * visible[:, 0] / visible[:, 2] + intrinsic[0, 2]
    y = intrinsic[1, 1] * visible[:, 1] / visible[:, 2] + intrinsic[1, 2]
    width, height = camera["resolution_wh"]
    box = [
        int(np.clip(np.floor(x.min()), 0, width - 1)),
        int(np.clip(np.floor(y.min()), 0, height - 1)),
        int(np.clip(np.ceil(x.max()), 0, width - 1)),
        int(np.clip(np.ceil(y.max()), 0, height - 1)),
    ]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("MesaTask target projects to an empty RGB box")
    return box


def prepare(
    raw_root: Path,
    output_root: Path,
    context_id: str,
    spec: dict,
    scene_record: dict,
    manifest_root: Path,
) -> None:
    source_dir = raw_root / "Layout_info" / spec["layout"]
    layout = json.loads((source_dir / "layout.json").read_text())
    target = next(item for item in layout["objects"] if item["retrieved_uid"] == spec["uid"])
    scene_path = resolve(manifest_root, scene_record["scene_glb"])
    if not scene_path.is_file():
        raise FileNotFoundError(f"3D-FRONT scene GLB not found: {scene_path}")
    if scene_record.get("scene_units") != "meters":
        raise ValueError(f"{context_id}: scene_units must explicitly be 'meters'")
    support_surface = str(scene_record["support_surface_id"])
    source_scene = trimesh.load(scene_path, force="scene", process=False)
    if support_surface not in source_scene.graph.nodes and support_surface not in source_scene.geometry:
        raise ValueError(
            f"{context_id}: support_surface_id {support_surface!r} is not a node or geometry "
            "in the supplied 3D-FRONT scene"
        )
    layout_to_world, frame_rotation = _layout_transform(scene_record)
    camera = scene_record["camera"]
    width, height = map(int, camera["resolution_wh"])
    if width % 32 or height % 32:
        raise ValueError(
            f"{context_id}: calibrated render dimensions must be divisible by 32 "
            "so the Qwen image processor can preserve them exactly"
        )
    output = output_root / context_id
    output.mkdir(parents=True, exist_ok=False)

    combined_scene = trimesh.Scene()
    sampled_geometry = _copy_front_scene(source_scene, combined_scene)
    target_pose = None
    target_mesh_info = None
    for object_index, item in enumerate(layout["objects"]):
        uid = item.get("retrieved_uid")
        if not uid:
            raise ValueError(f"MesaTask object {item.get('instance')} has no retrieved_uid")
        asset_path = raw_root / "selected_assets" / f"{uid}.glb"
        object_mesh, raw_center, raw_extents = canonical_mesh(asset_path)
        position, rotation, size = _mesh_pose(item, layout_to_world, frame_rotation)
        transform = _pose_matrix(position, rotation, size)
        node_name = f"mesatask_object_{object_index:03d}_{uid}"
        sampled_geometry.extend(
            _add_object_scene(combined_scene, object_mesh, transform, node_name)
        )
        if uid == spec["uid"]:
            target_pose = {
                "position_m": position.tolist(),
                "rotation_object_to_world": rotation.tolist(),
                "size_m": size.tolist(),
            }
            target_mesh_info = (object_mesh, raw_center, raw_extents)
    if target_pose is None or target_mesh_info is None:
        raise ValueError(f"target asset {spec['uid']} is absent from MesaTask layout")

    rgb, _ = _pyrender_scene(combined_scene, camera)
    rgb_path = output / "rgb.png"
    Image.fromarray(rgb, "RGB").save(rgb_path, optimize=True)
    scene_geometry_path = output / "combined_scene.glb"
    combined_scene.export(scene_geometry_path)
    cloud = _sample_scene_cloud(sampled_geometry, int(scene_record.get("point_count", 250000)), 20260928)
    np.save(output / "point_cloud_world_m.npy", cloud)
    (output / "camera.json").write_text(json.dumps(camera, indent=2) + "\n")
    (output / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")

    target_bbox = _project_target_box(target_pose, camera)
    mask = Image.new("L", (int(camera["resolution_wh"][0]), int(camera["resolution_wh"][1])), 0)
    ImageDraw.Draw(mask).rectangle(target_bbox, fill=255)
    mask.save(output / "target_mask.png")
    target_mesh, raw_center, raw_extents = target_mesh_info
    target_mesh.export(output / "mesh.glb")
    record = {
        "context_id": context_id,
        "source": "populated_3d_front",
        "source_identifier": f"MesaTask-10K/{spec['layout']}/{target['instance']}+{scene_record['scene_identifier']}",
        "scene_identifier": scene_record["scene_identifier"],
        "support_surface_id": support_surface,
        "layout_to_world": layout_to_world.tolist(),
        "scene_geometry_path": "combined_scene.glb",
        "target_description": spec["description"],
        "target_category": target["instance"].split("_", 2)[1],
        "target_bbox_xyxy": target_bbox,
        "mesh_path": "mesh.glb",
        "rgb_paths": ["rgb.png"],
        "camera_path": "camera.json",
        "point_cloud_path": "point_cloud_world_m.npy",
        "layout_path": "layout.json",
        "target_pose": target_pose,
        "coordinate_convention": "3D-FRONT world in meters; MesaTask tabletop coordinates transformed by layout_to_world",
        "units": "meters",
        "geometry_provenance": {
            "rgb_and_scene_geometry": str(scene_path),
            "rgb_render": "combined 3D-FRONT scene plus every MesaTask layout asset using the calibrated virtual camera",
            "scene_point_cloud": "surface samples from the complete combined triangle geometry; RGB assigned by calibrated camera projection at render time",
            "layout_transform": "explicit rigid tabletop-to-world transform from the scene manifest",
            "raw_target_mesh_bbox_center": raw_center.tolist(),
            "raw_target_mesh_extents": raw_extents.tolist(),
            "layout_asset_count": len(layout["objects"]),
        },
    }
    (output / "context.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"context": context_id, "scene": record["scene_identifier"], "objects": len(layout["objects"]), "points": len(cloud)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path, help="MesaTask release root")
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--scene-manifest",
        type=Path,
        required=True,
        help="JSON with contexts.front_01/front_02 scene_glb, scene_identifier, scene_units, support_surface_id, layout_to_world and calibrated camera",
    )
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = args.scene_manifest.resolve()
    scene_manifest = json.loads(manifest_path.read_text())
    contexts = scene_manifest.get("contexts", {})
    if set(contexts) != set(CONTEXTS):
        raise ValueError(f"scene manifest contexts must be exactly {sorted(CONTEXTS)}")
    output_root.mkdir(parents=True, exist_ok=True)
    for context_id, spec in CONTEXTS.items():
        prepare(
            raw_root,
            output_root,
            context_id,
            spec,
            contexts[context_id],
            manifest_path.parent,
        )


if __name__ == "__main__":
    main()
