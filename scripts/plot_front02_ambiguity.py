#!/usr/bin/env python3
"""Visualize the near-indistinguishable front_02 trajectory-08/09 inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/overfit50"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/front_02_08_09_ambiguity.png")
    )
    args = parser.parse_args()

    image_08 = load_rgb(args.dataset_root / "trajectories/front_02/08/step_00.png")
    image_09 = load_rgb(args.dataset_root / "trajectories/front_02/09/step_00.png")
    abs_diff = np.abs(image_08.astype(np.int16) - image_09.astype(np.int16))
    diff_strength = abs_diff.max(axis=2)
    ys, xs = np.where(diff_strength > 10)
    if not len(xs):
        raise RuntimeError("the selected images have no visible pixel differences")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad = 55
    left, right = max(0, x0 - pad), min(image_08.shape[1], x1 + pad + 1)
    top, bottom = max(0, y0 - pad), min(image_08.shape[0], y1 + pad + 1)
    crop_08 = image_08[top:bottom, left:right]
    crop_09 = image_09[top:bottom, left:right]
    crop_diff = diff_strength[top:bottom, left:right]
    changed_pixels = int(np.count_nonzero(diff_strength > 10))
    total_pixels = int(diff_strength.size)
    mean_abs = float(abs_diff.mean())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(
        "Why front_02_08 and front_02_09 compete during overfitting\n"
        "Different target actions, but only a tiny current-mesh region changes in the input",
        fontsize=16,
        fontweight="bold",
    )
    case_titles = [
        "Case 08 input\ngold: rotate (-3.57, 30.88, 36.44)",
        "Case 09 input\ngold: rotate (30.04, 2.80, -57.43)",
    ]
    for ax, image, title in zip(axes[0, :2], (image_08, image_09), case_titles):
        ax.imshow(image)
        ax.add_patch(
            Rectangle((left, top), right - left, bottom - top,
                      fill=False, edgecolor="#ff3131", linewidth=3)
        )
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.02,
        0.88,
        "The red box is the only visible\n"
        "difference between the images.\n\n"
        f"Changed pixels (max RGB delta > 10):\n"
        f"{changed_pixels} / {total_pixels} = {changed_pixels / total_pixels:.4%}\n\n"
        f"Mean absolute RGB difference:\n{mean_abs:.4f} on a 0–255 scale\n\n"
        "At model resolution, the pen/current mesh\n"
        "occupies only a few pixels. The two large\n"
        "pose-update strings therefore compete.",
        va="top",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#fff4e5", "edgecolor": "#e67e22"},
    )

    for ax, image, title in zip(
        axes[1, :2],
        (crop_08, crop_09),
        ("Case 08: enlarged red-box crop", "Case 09: enlarged red-box crop"),
    ):
        ax.imshow(image, interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    heatmap = axes[1, 2].imshow(crop_diff, cmap="magma", vmin=0, vmax=255, interpolation="nearest")
    axes[1, 2].set_title("Absolute RGB difference in the crop", fontsize=12)
    axes[1, 2].axis("off")
    fig.colorbar(heatmap, ax=axes[1, 2], fraction=0.046, pad=0.04, label="max RGB delta")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
