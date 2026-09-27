#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_TARGET_ID = "5611597c-1982-4ada-8637-c7779f6bd7e5"
DEFAULT_TIMESTAMPS = (
    "2501213518458",
    "2501113559250",
    "2501013600708",
    "2500913641791",
)


def load_json(path: Path):
    return json.loads(path.read_text())


def unproject(depth_m: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(depth_m.shape)
    z = depth_m
    x = (cols - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (rows - intrinsic[1, 2]) * z / intrinsic[1, 1]
    return np.stack((x, y, z), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-id", default=DEFAULT_TARGET_ID)
    parser.add_argument("--timestamps", nargs=4, default=DEFAULT_TIMESTAMPS)
    parser.add_argument("--point-stride", type=int, default=2)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    world_instances = load_json(args.capture_root / "world.gt/instances.json")
    target = next(item for item in world_instances if item["id"] == args.target_id)
    (args.output / "source_instance.json").write_text(json.dumps(target, indent=2))

    point_clouds: list[np.ndarray] = []
    view_records: list[dict] = []
    for view_index, timestamp in enumerate(args.timestamps):
        wide_dir = args.capture_root / f"{timestamp}.wide"
        gt_dir = args.capture_root / f"{timestamp}.gt"
        per_frame = load_json(wide_dir / "instances.json")
        instance = next(item for item in per_frame if item["id"] == args.target_id)

        rgb = Image.open(wide_dir / "image.png").convert("RGB")
        bbox = [float(value) for value in instance["box_2d_rend"]]
        annotated = rgb.copy()
        ImageDraw.Draw(annotated).rectangle(bbox, outline=(255, 215, 0), width=5)
        target_mask = Image.new("L", rgb.size, 0)
        ImageDraw.Draw(target_mask).rectangle(bbox, fill=255)
        rgb_name = f"view_{view_index:02d}_rgb.png"
        annotated_name = f"view_{view_index:02d}_target.png"
        mask_name = f"view_{view_index:02d}_mask.png"
        rgb.save(args.output / rgb_name)
        annotated.save(args.output / annotated_name)
        target_mask.save(args.output / mask_name)

        depth_mm = np.asarray(Image.open(gt_dir / "depth.png"), dtype=np.float64)
        intrinsic = np.asarray(load_json(gt_dir / "depth/K.json"), dtype=np.float64).reshape(3, 3)
        camera_to_world = np.asarray(load_json(gt_dir / "RT.json"), dtype=np.float64).reshape(4, 4)
        points_camera = unproject(depth_mm / 1000.0, intrinsic)
        valid = (depth_mm > 0) & (depth_mm < 10000)
        points_camera = points_camera[:: args.point_stride, :: args.point_stride]
        valid = valid[:: args.point_stride, :: args.point_stride]
        homogeneous = np.concatenate(
            (points_camera[valid], np.ones((int(valid.sum()), 1))), axis=1
        )
        points_world = (camera_to_world @ homogeneous.T).T[:, :3]
        point_clouds.append(points_world.astype(np.float32))

        depth_name = f"view_{view_index:02d}_depth_mm.png"
        depth_output = args.output / depth_name
        if depth_output.exists():
            depth_output.chmod(0o664)
        shutil.copyfile(gt_dir / "depth.png", depth_output)
        view_records.append(
            {
                "timestamp": timestamp,
                "rgb_path": rgb_name,
                "annotated_rgb_path": annotated_name,
                "target_mask_path": mask_name,
                "depth_path": depth_name,
                "target_bbox_xyxy": bbox,
                "rgb_intrinsic": load_json(wide_dir / "image/K.json"),
                "depth_intrinsic": intrinsic.tolist(),
                "camera_to_world": camera_to_world.tolist(),
            }
        )

    merged = np.concatenate(point_clouds, axis=0)
    np.save(args.output / "point_cloud_world_m.npy", merged)
    context = {
        "context_id": "ca1m_01",
        "source": "ca1m_objects",
        "source_capture": args.capture_root.name,
        "target_id": args.target_id,
        "target_category": target["category"],
        "target_description": target.get("caption", target["category"]),
        "target_box_world": {
            "position_m": target["position"],
            "rotation_object_to_world": target["R"],
            "size_m": target["scale"],
        },
        "views": view_records,
        "point_cloud_path": "point_cloud_world_m.npy",
        "coordinate_convention": "CA-1M laser-scanner world; camera_to_world from gt/RT.json",
        "units": "meters",
        "asset_status": "SAM 3D mesh and manual alignment pending",
    }
    (args.output / "context.json").write_text(json.dumps(context, indent=2))
    print(json.dumps({"views": len(view_records), "points": len(merged), "output": str(args.output)}))


if __name__ == "__main__":
    main()
