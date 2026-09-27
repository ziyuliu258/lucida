#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lucida_mini.schema import ContextRecord, DatasetManifest, TrajectoryRecord


CONTEXT_IDS = ("front_01", "front_02", "foundationpose_01", "foundationpose_02", "ca1m_01")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    contexts = []
    trajectories = []
    for context_id in CONTEXT_IDS:
        context_dir = args.dataset_root / "contexts" / context_id
        raw = json.loads((context_dir / "context.json").read_text())
        prefix = Path("contexts") / context_id
        raw["mesh_path"] = str(prefix / raw["mesh_path"])
        raw["camera_path"] = str(prefix / raw["camera_path"])
        raw["point_cloud_path"] = str(prefix / raw["point_cloud_path"])
        raw["rgb_paths"] = [str(prefix / item) for item in raw["rgb_paths"]]
        raw["depth_paths"] = [str(prefix / item) for item in raw.get("depth_paths", [])]
        contexts.append(ContextRecord.model_validate(raw))
        trajectory_dir = args.dataset_root / "trajectories" / context_id
        paths = sorted(trajectory_dir.glob("*/trajectory.json"))
        if len(paths) != 10:
            raise ValueError(f"{context_id}: expected 10 trajectories, found {len(paths)}")
        trajectories.extend(TrajectoryRecord.model_validate_json(path.read_text()) for path in paths)
    manifest = DatasetManifest(
        name="gizmoact_overfit50", frozen=args.freeze,
        contexts=contexts, trajectories=trajectories,
    )
    output = args.dataset_root / "manifest.json"
    output.write_text(manifest.model_dump_json(indent=2))
    print(json.dumps({"output": str(output), "frozen": args.freeze, "contexts": 5, "trajectories": 50}))


if __name__ == "__main__":
    main()
