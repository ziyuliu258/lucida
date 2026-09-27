#!/usr/bin/env python3
"""Run the exact multimodal collator over every supervised fixed-set turn."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml
from transformers import AutoProcessor

from lucida_mini.schema import DatasetManifest
from train_sft import GizmoCollator, TurnDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    train = config["training"]
    model = config["model"]
    processor = AutoProcessor.from_pretrained(model["name_or_path"])
    processor.image_processor.max_pixels = model["vision_max_pixels"]
    processor.image_processor.min_pixels = model["vision_min_pixels"]
    manifest = DatasetManifest.model_validate_json(args.manifest.read_text())
    dataset = TurnDataset(manifest, args.dataset_root, config.get("action_weighting"))
    collator = GizmoCollator(processor, train["max_sequence_length"], config.get("action_weighting"))

    rows = []
    maximum = 0
    for index, example in enumerate(dataset):
        batch = collator([example])
        length = int(batch["input_ids"].shape[1])
        maximum = max(maximum, length)
        rows.append({
            "trajectory_id": example["trajectory_id"],
            "step": example["step"],
            "input_tokens": length,
            "max_sequence_length": train["max_sequence_length"],
            "remaining_tokens": train["max_sequence_length"] - length,
        })
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            print(f"audited {index + 1}/{len(dataset)} turns; longest={maximum}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "examples": len(rows),
        "max_sequence_length": train["max_sequence_length"],
        "longest_input_tokens": maximum,
        "minimum_remaining_tokens": min(row["remaining_tokens"] for row in rows),
        "all_actions_complete": True,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
