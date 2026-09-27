#!/usr/bin/env python3
"""Plot one fixed-dataset overfit report across successive SFT phases.

The regular :mod:`plot_training` utility plots a single training run.  This
script is for the final fixed-50-trajectory report, where later SFT phases
start from an earlier adapter but reset their local optimizer step.  It offsets
the local steps to make a cumulative x-axis, retains the phase boundaries, and
labels action-weighted objectives explicitly so their raw loss scale is never
mistaken for ordinary token cross entropy.

Each ``--phase`` value is ``LABEL=OUTPUT_DIR``.  An output directory must have
``train_metrics.csv`` and may have completed ``eval-<step>/summary.csv``
subdirectories (or an ``eval_metrics.csv`` fallback).  Incomplete evaluation
directories are ignored, which makes it safe to use while a later phase is
still training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

# This is a batch artifact generator; make it work on a headless training node.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EVAL_DIRECTORY = re.compile(r"eval-(\d+)")
METRIC_PANELS = (
    ("add_sb_m", "ADD-SB (m)", False),
    ("add_sb_fraction", "ADD-SB / diameter", False),
    ("iou_3d", "oriented 3D IoU", True),
    ("rotation_error_deg", "rotation error (deg)", False),
    ("add_sb_at_0.10", "ADD-SB@0.10", True),
    ("add_sb_at_0.05", "ADD-SB@0.05", True),
    ("add_sb_at_0.01", "ADD-SB@0.01", True),
    ("rotation_at_5deg", "Rot.@5°", True),
    ("valid_action_rate", "valid GizmoAct", True),
    ("stopped", "explicit stop", True),
    ("actions", "model actions / rollout", False),
    ("closed_loop_success", "closed-loop success", True),
)
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728")


def parse_phase_spec(value: str) -> tuple[str, Path]:
    """Parse a ``LABEL=OUTPUT_DIR`` command-line value."""
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            f"phase must be LABEL=OUTPUT_DIR, got {value!r}"
        )
    return label.strip(), Path(raw_path.strip())


def read_numeric_csv(path: Path) -> list[dict[str, float]]:
    """Read a nonempty, fully numeric CSV with useful source-specific errors."""
    if not path.is_file():
        raise FileNotFoundError(f"missing CSV: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"CSV has no rows: {path}")

    rows: list[dict[str, float]] = []
    for row_index, raw in enumerate(raw_rows, start=2):
        parsed: dict[str, float] = {}
        for column in reader.fieldnames:
            value = raw.get(column)
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"non-numeric {column}={value!r} at {path}:{row_index}"
                ) from exc
            if not math.isfinite(number):
                raise ValueError(
                    f"non-finite {column}={value!r} at {path}:{row_index}"
                )
            parsed[column] = number
        rows.append(parsed)
    return rows


def require_steps(rows: list[dict[str, float]], source: Path) -> list[int]:
    """Validate a strictly increasing integer ``step`` column and return it."""
    if not rows or "step" not in rows[0]:
        raise ValueError(f"missing step column: {source}")
    steps: list[int] = []
    for row in rows:
        value = row["step"]
        if not value.is_integer() or value < 1:
            raise ValueError(f"step must be a positive integer in {source}: {value!r}")
        steps.append(int(value))
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError(f"steps must be strictly increasing: {source}")
    return steps


def collect_evaluations(output_dir: Path) -> list[dict[str, float]]:
    """Read complete closed-loop summaries in local-step order.

    Direct ``eval-<step>/summary.csv`` files are authoritative.  The
    ``eval_metrics.csv`` fallback supports an older run where summaries have
    already been aggregated and the directories are unavailable.
    """
    summaries: list[tuple[int, Path]] = []
    for path in output_dir.glob("eval-*/summary.csv"):
        match = EVAL_DIRECTORY.fullmatch(path.parent.name)
        if match is not None:
            summaries.append((int(match.group(1)), path))
    summaries.sort(key=lambda item: item[0])

    if summaries:
        observed: set[int] = set()
        rows: list[dict[str, float]] = []
        for directory_step, path in summaries:
            if directory_step in observed:
                raise ValueError(f"duplicate evaluation step under {output_dir}: {directory_step}")
            observed.add(directory_step)
            summary_rows = read_numeric_csv(path)
            if len(summary_rows) != 1:
                raise ValueError(f"summary must contain one row: {path}")
            row = summary_rows[0]
            if "step" not in row or not row["step"].is_integer() or int(row["step"]) != directory_step:
                raise ValueError(f"summary step does not match {path.parent.name}: {path}")
            rows.append(row)
        return rows

    aggregate = output_dir / "eval_metrics.csv"
    if not aggregate.is_file():
        return []
    rows = read_numeric_csv(aggregate)
    require_steps(rows, aggregate)
    return rows


def config_is_action_weighted(output_dir: Path) -> bool:
    """Return whether a completed run's resolved config enabled action weighting."""
    path = output_dir / "resolved_config.json"
    if not path.is_file():
        return False
    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config: {path}") from exc
    weighting = config.get("action_weighting", {}) if isinstance(config, dict) else {}
    return isinstance(weighting, dict) and bool(weighting.get("enabled", False))


def load_phases(
    phase_specs: list[tuple[str, Path]], weighted_labels: set[str] | None = None
) -> list[dict[str, Any]]:
    """Load phase logs and apply non-overlapping cumulative-step offsets."""
    if not phase_specs:
        raise ValueError("at least one phase is required")
    weighted_labels = weighted_labels or set()
    labels = [label for label, _ in phase_specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"phase labels must be unique: {labels}")
    unknown_weighted = weighted_labels - set(labels)
    if unknown_weighted:
        raise ValueError(f"weighted phase labels were not supplied: {sorted(unknown_weighted)}")

    offset = 0
    phases: list[dict[str, Any]] = []
    for index, (label, output_dir) in enumerate(phase_specs):
        train_path = output_dir / "train_metrics.csv"
        train_rows = read_numeric_csv(train_path)
        if "loss" not in train_rows[0]:
            raise ValueError(f"training log has no loss column: {train_path}")
        train_steps = require_steps(train_rows, train_path)
        metric_rows = collect_evaluations(output_dir)

        weighted = label in weighted_labels or config_is_action_weighted(output_dir)
        phase = {
            "label": label,
            "directory": output_dir,
            "weighted": weighted,
            "color": COLORS[index % len(COLORS)],
            "offset": offset,
            "duration": max(train_steps),
            "train": train_rows,
            "metrics": metric_rows,
        }
        for row in train_rows:
            row["cumulative_step"] = row["step"] + offset
        for row in metric_rows:
            row["cumulative_step"] = row["step"] + offset
        phases.append(phase)
        offset += max(train_steps)
    return phases


def moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return a trailing moving mean without hiding short phase logs."""
    if window <= 1 or len(values) < window:
        return values
    result = np.full_like(values, np.nan, dtype=float)
    result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def phase_objective_label(phase: dict[str, Any]) -> str:
    if phase["weighted"]:
        return f'{phase["label"]}: weighted action loss'
    return f'{phase["label"]}: token cross-entropy'


def add_phase_boundaries(axis: plt.Axes, phases: list[dict[str, Any]], *, annotate: bool) -> None:
    """Shade each phase and optionally place its objective above the loss axis."""
    for phase in phases:
        start = phase["offset"]
        end = start + phase["duration"]
        axis.axvspan(start, end, color=phase["color"], alpha=0.045, linewidth=0)
        if start:
            axis.axvline(start, color="0.55", linewidth=0.8, linestyle="--", alpha=0.7)
        if annotate:
            axis.text(
                (start + end) / 2,
                0.98,
                phase_objective_label(phase),
                color=phase["color"],
                ha="center",
                va="top",
                fontsize=8,
                transform=axis.get_xaxis_transform(),
            )


def plot_summary(
    phases: list[dict[str, Any]], output: Path, smooth_window: int = 25
) -> None:
    """Write the combined loss and available closed-loop metric figure as PNG."""
    if smooth_window < 1:
        raise ValueError("smooth window must be positive")
    available_panels = [
        panel
        for panel in METRIC_PANELS
        if any(panel[0] in row for phase in phases for row in phase["metrics"])
    ]
    columns = 4
    metric_rows = math.ceil(len(available_panels) / columns)
    figure = plt.figure(figsize=(18, 4.4 + 3.45 * metric_rows))
    grid = figure.add_gridspec(1 + metric_rows, columns, hspace=0.43, wspace=0.28)
    loss_axis = figure.add_subplot(grid[0, :])

    for phase in phases:
        train = phase["train"]
        steps = np.asarray([row["cumulative_step"] for row in train])
        loss = np.asarray([row["loss"] for row in train])
        loss_axis.plot(
            steps,
            loss,
            color=phase["color"],
            alpha=0.14,
            linewidth=0.8,
        )
        loss_axis.plot(
            steps,
            moving_mean(loss, smooth_window),
            color=phase["color"],
            linewidth=1.75,
            label=phase_objective_label(phase),
        )
    add_phase_boundaries(loss_axis, phases, annotate=False)
    loss_axis.set(
        title=(
            "Fixed-50-trajectory SFT loss "
            "(weighted action-loss values are a different objective from token CE)\n"
            "Closed-loop metrics use completed evaluation summaries only"
        ),
        ylabel="logged training loss",
    )
    loss_axis.grid(alpha=0.22)
    loss_axis.legend(loc="upper right", fontsize=9)

    for index, (key, title, bounded) in enumerate(available_panels):
        axis = figure.add_subplot(grid[1 + index // columns, index % columns])
        for phase in phases:
            rows = [row for row in phase["metrics"] if key in row]
            if not rows:
                continue
            axis.plot(
                [row["cumulative_step"] for row in rows],
                [row[key] for row in rows],
                color=phase["color"],
                marker="o",
                markersize=3.5,
                linewidth=1.35,
            )
        add_phase_boundaries(axis, phases, annotate=False)
        axis.set_title(title, fontsize=10)
        if bounded:
            axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.22)

    figure.supxlabel("cumulative optimizer step (phase-local steps offset)")
    output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        type=parse_phase_spec,
        action="append",
        required=True,
        metavar="LABEL=OUTPUT_DIR",
        help="One phase, in chronological order. Repeat for each phase.",
    )
    parser.add_argument(
        "--weighted-phase",
        action="append",
        default=[],
        metavar="LABEL",
        help="Explicitly mark a phase as action-weighted when no resolved config exists yet.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Final PNG path.")
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()

    phases = load_phases(args.phase, set(args.weighted_phase))
    plot_summary(phases, args.output, args.smooth_window)
    metric_points = sum(len(phase["metrics"]) for phase in phases)
    print(
        f"wrote {args.output.with_suffix('.png')} from {len(phases)} phases "
        f"and {metric_points} completed closed-loop summaries"
    )


if __name__ == "__main__":
    main()
