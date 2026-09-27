#!/usr/bin/env python3
"""Combine closed-loop evaluation summaries into one plotting CSV.

``plot_training.py`` expects one row per checkpoint.  Closed-loop evaluation
writes one self-contained ``eval-<step>/summary.csv`` per checkpoint instead,
so this utility validates those summaries and writes their step-sorted union.
It intentionally does not import or modify the training/evaluation code.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


METRIC_COLUMNS = (
    "step",
    "actions",
    "valid_action_rate",
    "stopped",
    "add_sb_m",
    "object_diameter_m",
    "add_sb_fraction",
    "add_sb_at_0.10",
    "add_sb_at_0.05",
    "add_sb_at_0.01",
    "iou_3d",
    "rotation_error_deg",
    "rotation_at_5deg",
    "closed_loop_success",
)

_STEP_DIRECTORY = re.compile(r"eval-(\d+)")
_UNIT_INTERVAL_COLUMNS = {
    "valid_action_rate",
    "stopped",
    "add_sb_at_0.10",
    "add_sb_at_0.05",
    "add_sb_at_0.01",
    "iou_3d",
    "rotation_at_5deg",
    "closed_loop_success",
}
_NONNEGATIVE_COLUMNS = {
    "actions",
    "add_sb_m",
    "add_sb_fraction",
    "rotation_error_deg",
}


def _directory_step(directory: Path) -> int:
    match = _STEP_DIRECTORY.fullmatch(directory.name)
    if match is None:
        raise ValueError(f"evaluation directory must be named eval-<step>: {directory}")
    return int(match.group(1))


def _read_single_summary(path: Path, directory_step: int) -> dict[str, float]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"summary has no header: {path}")
        missing = [column for column in METRIC_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"summary is missing required columns {missing}: {path}")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"summary must contain exactly one data row, got {len(rows)}: {path}")

    row: dict[str, float] = {}
    for column in METRIC_COLUMNS:
        raw = rows[0][column]
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric {column}={raw!r}: {path}") from exc
        if not math.isfinite(value):
            raise ValueError(f"non-finite {column}={raw!r}: {path}")
        row[column] = value

    step = row["step"]
    if not step.is_integer() or int(step) != directory_step:
        raise ValueError(
            f"summary step {step!r} does not match eval-{directory_step}: {path}"
        )
    row["step"] = int(step)
    if any(row[column] < 0.0 for column in _NONNEGATIVE_COLUMNS):
        raise ValueError(f"summary has a negative error/count metric: {path}")
    if row["object_diameter_m"] <= 0.0:
        raise ValueError(f"object_diameter_m must be positive: {path}")
    out_of_range = [
        column for column in _UNIT_INTERVAL_COLUMNS
        if not 0.0 <= row[column] <= 1.0
    ]
    if out_of_range:
        raise ValueError(f"summary has metrics outside [0, 1] {out_of_range}: {path}")
    return row


def collect_summaries(input_dir: Path) -> list[dict[str, float]]:
    """Read validated ``eval-<step>/summary.csv`` files in numerical order."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"evaluation root does not exist: {input_dir}")
    directories = [
        path for path in input_dir.iterdir()
        if path.is_dir() and _STEP_DIRECTORY.fullmatch(path.name)
    ]
    directories.sort(key=_directory_step)
    if not directories:
        raise FileNotFoundError(f"no eval-<step> directories under: {input_dir}")

    rows: list[dict[str, float]] = []
    observed_steps: set[int] = set()
    for directory in directories:
        step = _directory_step(directory)
        summary = directory / "summary.csv"
        if not summary.is_file():
            raise FileNotFoundError(f"missing evaluation summary: {summary}")
        row = _read_single_summary(summary, step)
        if step in observed_steps:
            raise ValueError(f"duplicate evaluation step {step} under: {input_dir}")
        observed_steps.add(step)
        rows.append(row)
    return rows


def write_metrics(rows: list[dict[str, float]], output_csv: Path) -> None:
    """Write deterministic, plot_training-compatible metric rows."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing eval-<step>/summary.csv subdirectories.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output file; defaults to <input-dir>/eval_metrics.csv.",
    )
    args = parser.parse_args()

    rows = collect_summaries(args.input_dir)
    output_csv = args.output_csv or args.input_dir / "eval_metrics.csv"
    write_metrics(rows, output_csv)
    print(f"wrote {len(rows)} evaluation rows to {output_csv}")


if __name__ == "__main__":
    main()
