#!/usr/bin/env python3
"""Plot reproducible loss and closed-loop metric curves from CSV logs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty log: {path}")
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    result = np.full_like(values, np.nan, dtype=float)
    kernel = np.ones(window) / window
    result[window - 1 :] = np.convolve(values, kernel, mode="valid")
    return result


def save(fig, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--teacher-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = read_csv(args.train_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train["step"], train["loss"], alpha=0.25, label="training loss")
    ax.plot(
        train["step"], smooth(train["loss"], args.smooth_window),
        label=f"moving mean ({args.smooth_window})",
    )
    if "full_train_loss" in train:
        mask = np.isfinite(train["full_train_loss"])
        ax.plot(train["step"][mask], train["full_train_loss"][mask], "o-", label="full 50")
    if args.teacher_csv:
        teacher = read_csv(args.teacher_csv)
        ax.plot(
            teacher["step"],
            teacher["full_train_loss"],
            "o-",
            label="full-set teacher-forced token CE",
        )
    ax.set(xlabel="optimizer step", ylabel="cross entropy", title="SFT overfit loss")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, args.output_dir / "loss_curve")

    metrics = read_csv(args.eval_csv)
    panels = [
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
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True)
    for ax, (key, label, bounded) in zip(axes.flat, panels):
        ax.plot(metrics["step"], metrics[key], marker="o", markersize=3)
        ax.set_title(label)
        if bounded:
            ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("optimizer step")
    save(fig, args.output_dir / "metric_curves")


if __name__ == "__main__":
    main()
