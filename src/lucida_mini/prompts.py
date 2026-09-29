from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You control a 9-DoF GizmoAct object editor. Inspect the current observation "
    "and action history. Output exactly one valid GizmoAct XML action and no other text."
)


def _value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


@lru_cache(maxsize=128)
def _camera_views(path: str) -> tuple[dict, ...]:
    camera = json.loads(Path(path).read_text())
    return tuple(camera.get("views", ()))


def _camera_path(context: Any, context_root: Path) -> Path:
    relative = Path(_value(context, "camera_path", "camera.json"))
    candidate = relative if relative.is_absolute() else context_root / relative
    if candidate.is_file():
        return candidate
    local = context_root / "camera.json"
    if local.is_file():
        return local
    raise FileNotFoundError(f"camera calibration not found: {candidate}")


def frame_index_for_turn(context: Any, turn: Any, context_root: Path) -> int | None:
    """Return the selected CA-1M source frame for this interaction turn."""
    if _value(context, "source") != "ca1m_objects":
        return None
    explicit = _value(turn, "observation_frame_index")
    views = _camera_views(str(_camera_path(context, context_root)))
    if not views:
        raise ValueError(f"CA-1M context has no calibrated views: {context_root}")
    index = int(explicit) if explicit is not None else int(_value(turn, "step", 0))
    return min(max(index, 0), len(views) - 1)


def instruction_for_turn(context: Any, turn: Any, context_root: Path) -> str:
    """Build the stable action request with this target's source-image cue."""
    description = str(_value(context, "target_description", "target object")).strip()
    bbox = _value(context, "target_bbox_xyxy")
    frame_index = frame_index_for_turn(context, turn, context_root)
    if frame_index is not None:
        views = _camera_views(str(_camera_path(context, context_root)))
        bbox = views[frame_index].get("target_bbox_xyxy", bbox)
    if bbox is None or len(bbox) != 4:
        raise ValueError(f"target cue requires a four-value image box for {_value(context, 'context_id')}")
    box = ", ".join(f"{float(value):.0f}" for value in bbox)
    return (
        "Adjustment target: 3D model.\n"
        f"Target object: {description}.\n"
        f"Target position: around box [{box}] in raw image.\n"
        "Predict the next GizmoAct action."
    )
