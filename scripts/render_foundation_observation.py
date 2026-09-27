#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lucida_mini.pose import Pose
from lucida_mini.render import render_foundation_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("context_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pose-json", type=Path)
    args = parser.parse_args()
    context = json.loads((args.context_dir / "context.json").read_text())
    record = json.loads(args.pose_json.read_text()) if args.pose_json else context["target_pose"]
    pose = Pose(
        position=np.asarray(record["position_m"]),
        rotation=np.asarray(record["rotation_object_to_world"]),
        size=np.asarray(record["size_m"]),
    )
    render_foundation_observation(args.context_dir, pose, args.output)


if __name__ == "__main__":
    main()
