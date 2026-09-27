#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


CONTEXTS = {
    "front_01": {
        "layout": "kitchen_counter/kitchen_counter_0813",
        "uid": "5f5fa09d1f6349f993f100abd77d8f80",
        "bbox": [331, 179, 414, 303],
        "description": "antique hand-crank coffee grinder",
    },
    "front_02": {
        "layout": "office_table/office_table_0907",
        "uid": "e3774ca0bdae459082f3bed37fa7260c",
        "bbox": [357, 337, 381, 368],
        "description": "green clip pen",
    },
}


def canonical_mesh(path: Path) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="scene", process=False)
    mesh = trimesh.util.concatenate(tuple(loaded.geometry.values())) if isinstance(loaded, trimesh.Scene) else loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    lower, upper = vertices.min(0), vertices.max(0)
    center, extents = (lower + upper) / 2, upper - lower
    mesh.vertices = (vertices - center) / extents
    return mesh, center, extents


def box_surface(center: np.ndarray, size: np.ndarray, count: int = 120) -> np.ndarray:
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 0.5, (count, 3)) * size
    face = rng.integers(0, 3, count)
    side = rng.choice([-0.5, 0.5], count)
    points[np.arange(count), face] = side * size[face]
    return points + center


def prepare(raw_root: Path, output_root: Path, context_id: str, spec: dict) -> None:
    source_dir = raw_root / "Layout_info" / spec["layout"]
    layout = json.loads((source_dir / "layout.json").read_text())
    target = next(o for o in layout["objects"] if o["retrieved_uid"] == spec["uid"])
    output = output_root / context_id
    output.mkdir(parents=True, exist_ok=True)

    mesh, raw_center, raw_extents = canonical_mesh(raw_root / "selected_assets" / f"{spec['uid']}.glb")
    mesh.export(output / "mesh.glb")
    shutil.copy2(source_dir / "front.png", output / "rgb.png")
    shutil.copy2(source_dir / "layout.json", output / "layout.json")

    size = np.asarray(target["size"], dtype=np.float64) / 100.0
    position = np.asarray(target["position"], dtype=np.float64) / 100.0
    quaternion = np.asarray(target["rotation"], dtype=np.float64)
    rotation = Rotation.from_quat(quaternion).as_matrix()
    points = []
    for item in layout["objects"]:
        points.append(box_surface(np.asarray(item["position"]) / 100.0, np.asarray(item["size"]) / 100.0))
    zone = np.asarray(layout["item_placement_zone"], dtype=np.float64) / 100.0
    gx, gy = np.meshgrid(np.linspace(zone[0], zone[1], 100), np.linspace(zone[2], zone[3], 40))
    points.append(np.column_stack((gx.ravel(), gy.ravel(), np.zeros(gx.size))))
    cloud = np.concatenate(points).astype(np.float32)
    np.save(output / "point_cloud_world_m.npy", cloud)

    image = Image.open(output / "rgb.png")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle(spec["bbox"], fill=255)
    mask.save(output / "target_mask.png")
    camera = {
        "type": "display_projection",
        "resolution_wh": list(image.size),
        "description": "Fixed oblique visualization anchored to the released MesaTask target rectangle; not sensor calibration.",
    }
    (output / "camera.json").write_text(json.dumps(camera, indent=2))
    record = {
        "context_id": context_id,
        "source": "populated_3d_front",
        "source_identifier": f"MesaTask-10K/{spec['layout']}/{target['instance']}",
        "target_description": spec["description"],
        "target_category": target["instance"].split("_", 2)[1],
        "target_bbox_xyxy": spec["bbox"],
        "mesh_path": "mesh.glb",
        "rgb_paths": ["rgb.png"],
        "camera_path": "camera.json",
        "point_cloud_path": "point_cloud_world_m.npy",
        "layout_path": "layout.json",
        "target_pose": {
            "position_m": position.tolist(),
            "rotation_object_to_world": rotation.tolist(),
            "size_m": size.tolist(),
        },
        "coordinate_convention": "MesaTask layout: X/Y tabletop, Z up; canonical mesh per-axis unit box",
        "units": "meters",
        "geometry_provenance": {
            "rgb_and_layout": "official MesaTask-10K release",
            "target_mesh": "official MesaTask asset GLB",
            "scene_point_cloud": "deterministic box surfaces from released layout because the release has no depth image",
            "display_camera": "fixed approximate projection used only to visualize metric pose edits",
            "raw_mesh_bbox_center": raw_center.tolist(),
            "raw_mesh_extents": raw_extents.tolist(),
        },
    }
    (output / "context.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"context": context_id, "target": target["instance"], "points": len(cloud)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    for context_id, spec in CONTEXTS.items():
        prepare(args.raw_root, args.output_root, context_id, spec)


if __name__ == "__main__":
    main()
