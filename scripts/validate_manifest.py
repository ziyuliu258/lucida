#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lucida_mini.schema import DatasetManifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()

    manifest = DatasetManifest.model_validate_json(args.manifest.read_text())
    if args.check_files:
        missing = manifest.validate_files(args.manifest.parent)
        if missing:
            print(json.dumps({"missing_files": [str(path) for path in missing]}, indent=2))
            raise SystemExit(1)
    print(
        json.dumps(
            {
                "name": manifest.name,
                "frozen": manifest.frozen,
                "contexts": len(manifest.contexts),
                "trajectories": len(manifest.trajectories),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
