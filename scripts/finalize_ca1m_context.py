#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("context_dir", type=Path)
    parser.add_argument("sam_mesh", type=Path)
    args = parser.parse_args()
    context = json.loads((args.context_dir / "context.json").read_text())
    loaded = trimesh.load(args.sam_mesh, force="scene", process=False)
    mesh = trimesh.util.concatenate(tuple(loaded.geometry.values())) if isinstance(loaded, trimesh.Scene) else loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    lower, upper = vertices.min(0), vertices.max(0)
    center, extents = (lower + upper) / 2, upper - lower
    if np.any(extents <= 0):
        raise ValueError("SAM mesh has a degenerate bounding box")
    mesh.vertices = (vertices - center) / extents
    mesh.export(args.context_dir / "mesh.glb")
    camera = {"views": [{
        "rgb_intrinsic": view["rgb_intrinsic"],
        "camera_to_world": view["camera_to_world"],
        "resolution_wh": [768, 1024],
    } for view in context["views"]]}
    (args.context_dir / "camera.json").write_text(json.dumps(camera, indent=2))
    box = context.pop("target_box_world")
    context.update({
        "source_identifier": f"CA-1M/{context['source_capture']}/{context['target_id']}",
        "target_bbox_xyxy": context["views"][0]["target_bbox_xyxy"],
        "mesh_path": "mesh.glb",
        "rgb_paths": [view["rgb_path"] for view in context["views"]],
        "depth_paths": [view["depth_path"] for view in context["views"]],
        "camera_path": "camera.json",
        "target_pose": box,
        "asset_status": "SAM 3D mesh generated and aligned to the CA-1M oriented box",
        "sam3d_alignment": {
            "method": "canonicalize generated mesh to a per-axis unit box, then apply the released CA-1M oriented box",
            "raw_mesh_bbox_center": center.tolist(),
            "raw_mesh_extents": extents.tolist(),
        },
    })
    (args.context_dir / "context.json").write_text(json.dumps(context, indent=2))
    print(json.dumps({"vertices": len(vertices), "faces": len(mesh.faces), "output": str(args.context_dir / 'mesh.glb')}))


if __name__ == "__main__":
    main()
