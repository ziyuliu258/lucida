#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


TARGETS = {
    "foundationpose_01": ("Dino_3", 13, 4),
    "foundationpose_02": ("UGG_Bailey_Button_Womens_Boots_Black_7", 14, 5),
}


def find_one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} below {root}, found {len(matches)}")
    return matches[0]


def prepare(scene_root: Path, models_root: Path, output_root: Path, context_id: str) -> None:
    object_name, instance_id, bbox_index = TARGETS[context_id]
    product = find_one(scene_root, "scene-*/RenderProduct_Replicator")
    state = json.loads((scene_root / "states.json").read_text())["objects"][object_name]
    camera = json.loads(find_one(product, "camera_params/*.json").read_text())
    rgb_path = find_one(product, "rgb/*.png")
    depth_path = find_one(product, "distance_to_image_plane/*.npy")
    segmentation_path = find_one(product, "instance_segmentation/*.png")
    bbox_path = find_one(product, "bounding_box_2d_tight/*.npy")

    mesh = trimesh.load(models_root / object_name / "meshes/model.obj", force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) / 2.0
    centered = vertices - center
    radius = float(np.linalg.norm(centered, axis=1).max())
    extents = upper - lower
    # Store a per-axis unit box canonical mesh. Combined with target size below,
    # this is identical to FoundationPose's centered unit-sphere normalization.
    mesh.vertices = centered / extents

    scaled_rotation = np.asarray(state["rotation_matrix"], dtype=np.float64)
    scene_scale = np.linalg.norm(scaled_rotation, axis=0)
    if not np.allclose(scene_scale, scene_scale[0], rtol=1e-5, atol=1e-8):
        raise ValueError(f"expected uniform released scene scale for {object_name}")
    rotation = scaled_rotation / scene_scale[None, :]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"invalid target rotation for {object_name}")
    size = extents / radius * scene_scale[0]
    position = np.asarray(state["translation"], dtype=np.float64)

    output = output_root / context_id
    output.mkdir(parents=True, exist_ok=True)
    mesh.export(output / "mesh.glb")
    shutil.copy2(rgb_path, output / "rgb.png")
    np.save(output / "depth_m.npy", np.load(depth_path))

    segmentation = np.asarray(Image.open(segmentation_path))
    Image.fromarray((segmentation == instance_id).astype(np.uint8) * 255).save(output / "target_mask.png")
    boxes = np.load(bbox_path)
    box = boxes[bbox_index]
    bbox = [int(box[name]) for name in ("x_min", "y_min", "x_max", "y_max")]

    width, height = camera["renderProductResolution"]
    projection = np.asarray(camera["cameraProjection"], dtype=np.float64).reshape(4, 4)
    view = np.asarray(camera["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    depth = np.load(depth_path)
    rows, cols = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0)
    x_ndc = 2.0 * (cols + 0.5) / width - 1.0
    y_ndc = 1.0 - 2.0 * (rows + 0.5) / height
    camera_points = np.stack(
        (-depth * x_ndc / projection[0, 0], -depth * y_ndc / projection[1, 1], -depth),
        axis=-1,
    )[valid][::2]
    homogeneous = np.column_stack((camera_points, np.ones(len(camera_points))))
    world = homogeneous @ np.linalg.inv(view)
    np.save(output / "point_cloud_world_m.npy", world[:, :3].astype(np.float32))

    camera_record = {
        "projection_row_major": projection.tolist(),
        "world_to_camera_row_major": view.tolist(),
        "resolution_wh": [width, height],
        "convention": "Omniverse row-vector camera, view then projection",
    }
    (output / "camera.json").write_text(json.dumps(camera_record, indent=2))
    record = {
        "context_id": context_id,
        "source": "foundationpose",
        "source_identifier": f"gso/1202363524/scene_00000000/{object_name}",
        "target_description": object_name.replace("_", " "),
        "target_category": "toy dinosaur" if object_name == "Dino_3" else "black boot",
        "target_bbox_xyxy": bbox,
        "mesh_path": "mesh.glb",
        "rgb_paths": ["rgb.png"],
        "depth_paths": ["depth_m.npy"],
        "camera_path": "camera.json",
        "point_cloud_path": "point_cloud_world_m.npy",
        "target_pose": {
            "position_m": position.tolist(),
            "rotation_object_to_world": rotation.tolist(),
            "size_m": size.tolist(),
        },
        "coordinate_convention": "FoundationPose Omniverse world; canonical mesh per-axis unit box",
        "units": "meters",
        "normalization": {
            "released_pipeline": "centered unit bounding sphere",
            "raw_mesh_bbox_center": center.tolist(),
            "raw_mesh_bounding_radius": radius,
            "raw_mesh_extents": extents.tolist(),
        },
    }
    (output / "context.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"context": context_id, "object": object_name, "bbox": bbox, "points": len(world)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_root", type=Path)
    parser.add_argument("models_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    for context_id in TARGETS:
        prepare(args.scene_root, args.models_root, args.output_root, context_id)


if __name__ == "__main__":
    main()
