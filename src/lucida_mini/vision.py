from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def image_paths_in_messages(messages: list[dict]) -> list[Path]:
    paths: list[Path] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image" and item.get("image"):
                paths.append(Path(item["image"]))
    return paths


def validate_image_files(
    paths: list[Path],
    max_pixels: int,
    patch_size: int = 16,
    spatial_merge_size: int = 2,
) -> list[tuple[int, int]]:
    """Reject stored views that need resizing before they reach the VLM."""
    factor = patch_size * spatial_merge_size
    sizes: list[tuple[int, int]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
        if width % factor or height % factor:
            raise ValueError(
                f"{path} is {width}x{height}; dimensions must be divisible by {factor} "
                "so Qwen can preserve the native pixel size"
            )
        if width * height > max_pixels:
            raise ValueError(
                f"{path} is {width}x{height} ({width * height} pixels), above "
                f"vision_max_pixels={max_pixels}; refusing silent downsampling"
            )
        sizes.append((width, height))
    return sizes


def validate_processed_image_grids(
    encoded: dict[str, Any],
    sizes: list[tuple[int, int]],
    patch_size: int = 16,
) -> None:
    """Verify the image processor kept every input image at its stored size."""
    grid = encoded.get("image_grid_thw")
    if grid is None:
        if sizes:
            raise ValueError("Qwen processor returned no image_grid_thw for image inputs")
        return
    rows = grid.detach().cpu().tolist() if hasattr(grid, "detach") else list(grid)
    if len(rows) != len(sizes):
        raise ValueError(
            f"Qwen processor returned {len(rows)} image grids for {len(sizes)} input images"
        )
    for index, (row, (width, height)) in enumerate(zip(rows, sizes)):
        expected = (1, height // patch_size, width // patch_size)
        actual = tuple(int(value) for value in row)
        if actual != expected:
            raise ValueError(
                f"Qwen resized image {index}: stored {width}x{height} expects grid "
                f"{expected}, processor returned {actual}"
            )
