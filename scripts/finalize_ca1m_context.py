#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from lucida_mini.pose import Pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("context_dir", type=Path)
    parser.add_argument("sam_mesh", type=Path)
    parser.add_argument(
        "--manual-alignment",
        type=Path,
        required=True,
        help="JSON containing target_pose plus annotator and method metadata for alignment to the LiDAR point cloud",
    )
    args = parser.parse_args()
    if any(
        (args.context_dir / name).exists()
        for name in ("mesh.glb", "camera.json", "manual_alignment.json")
    ):
        raise FileExistsError(
            f"refusing to overwrite finalized CA-1M context: {args.context_dir}"
        )
    context = json.loads((args.context_dir / "context.json").read_text())
    if context.get("source_split") != "train":
        raise ValueError("CA-1M overfit context must come from the recorded train split")
    alignment_path = args.manual_alignment.resolve()
    alignment = json.loads(alignment_path.read_text())
    if not alignment.get("annotator") or not alignment.get("method"):
        raise ValueError("manual alignment JSON must record both annotator and method")
    target_pose = alignment["target_pose"]
    # Validate the supplied manual pose before writing any finalized artifacts.
    Pose(
        position=np.asarray(target_pose["position_m"], dtype=np.float64),
        rotation=np.asarray(target_pose["rotation_object_to_world"], dtype=np.float64),
        size=np.asarray(target_pose["size_m"], dtype=np.float64),
    )
    loaded = trimesh.load(args.sam_mesh, force="scene", process=False)
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
    else:
        instances = [("sam3d_mesh", loaded.copy(), np.eye(4, dtype=np.float64))]
        world_vertices = [np.asarray(loaded.vertices, dtype=np.float64)]
    if not world_vertices or any(not len(vertices) for vertices in world_vertices):
        raise ValueError("SAM mesh contains no vertices")
    vertices = np.concatenate(world_vertices, axis=0)
    lower, upper = vertices.min(0), vertices.max(0)
    center, extents = (lower + upper) / 2, upper - lower
    if not np.isfinite(extents).all() or np.any(extents <= 1e-9):
        raise ValueError("SAM mesh has a degenerate bounding box")
    normalize = np.eye(4, dtype=np.float64)
    normalize[:3, :3] = np.diag(1.0 / extents)
    normalize[:3, 3] = -center / extents
    mesh_scene = trimesh.Scene()
    for index, (node, geometry, transform) in enumerate(instances):
        mesh_scene.add_geometry(
            geometry,
            geom_name=f"sam3d_geometry_{index:03d}",
            node_name=f"sam3d_{node}_{index:03d}",
            transform=normalize @ transform,
        )
    mesh_scene.export(args.context_dir / "mesh.glb")
    face_count = sum(len(geometry.faces) for _node, geometry, _transform in instances)
    camera_views = []
    for view in context["views"]:
        rgb_path = args.context_dir / view["rgb_path"]
        with Image.open(rgb_path) as image:
            actual_resolution = [image.width, image.height]
        resolution = [int(value) for value in view["resolution_wh"]]
        if actual_resolution != resolution:
            raise ValueError(
                f"CA-1M source frame {rgb_path} changed resolution: "
                f"recorded {resolution}, actual {actual_resolution}"
            )
        if any(value % 32 for value in resolution):
            raise ValueError(
                f"CA-1M native frame dimensions must be divisible by 32: {resolution}"
            )
        camera_views.append({
            "rgb_intrinsic": view["rgb_intrinsic"],
            "camera_to_world": view["camera_to_world"],
            "resolution_wh": resolution,
            "target_bbox_xyxy": view["target_bbox_xyxy"],
        })
    camera = {"views": camera_views}
    (args.context_dir / "camera.json").write_text(json.dumps(camera, indent=2))
    annotation_box = context.pop("target_box_world")
    context.update({
        "source_identifier": f"CA-1M/{context['source_capture']}/{context['target_id']}",
        "target_bbox_xyxy": context["views"][0]["target_bbox_xyxy"],
        "mesh_path": "mesh.glb",
        "rgb_paths": [view["rgb_path"] for view in context["views"]],
        "depth_paths": [view["depth_path"] for view in context["views"]],
        "camera_path": "camera.json",
        "target_pose": target_pose,
        "reference_annotation_box": annotation_box,
        "asset_status": "SAM 3D mesh manually aligned to the CA-1M LiDAR point cloud",
        "asset_alignment_method": "manual_lidar_pointcloud_alignment",
        "alignment_path": "manual_alignment.json",
        "alignment_provenance": {
            "file": "manual_alignment.json",
            "sha256": hashlib.sha256(alignment_path.read_bytes()).hexdigest(),
            "annotator": alignment.get("annotator"),
            "method": alignment["method"],
            "reference_point_cloud": "point_cloud_world_m.npy",
        },
        "sam3d_alignment": {
            "method": "canonicalize generated mesh to a per-axis unit box; use the recorded manual pose and metric size aligned to LiDAR",
            "raw_mesh_bbox_center": center.tolist(),
            "raw_mesh_extents": extents.tolist(),
        },
    })
    shutil.copy2(alignment_path, args.context_dir / "manual_alignment.json")
    (args.context_dir / "context.json").write_text(json.dumps(context, indent=2))
    print(json.dumps({"vertices": len(vertices), "faces": face_count, "output": str(args.context_dir / 'mesh.glb')}))


if __name__ == "__main__":
    main()
