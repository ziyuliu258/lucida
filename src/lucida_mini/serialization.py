from __future__ import annotations

import json

import numpy as np

from .actions import PermuteAxis, Stop, SwitchObs, UpdatePose


def _clean(value: float) -> float:
    result = round(float(value), 2)
    return 0.0 if result == -0.0 else result


def action_to_text(action) -> str:
    if isinstance(action, Stop):
        return "<stop>{}</stop>"
    if isinstance(action, SwitchObs):
        return "<switch_obs>permute_axis</switch_obs>"
    if isinstance(action, PermuteAxis):
        return f'<permute_axis>{json.dumps({"x": action.x, "z": action.z}, separators=(",", ":"))}</permute_axis>'
    if not isinstance(action, UpdatePose):
        raise TypeError(f"unsupported action type: {type(action)!r}")
    payload: dict[str, dict[str, float | str]] = {}
    if action.rotate_zxy_deg is not None:
        z, x, y = np.asarray(action.rotate_zxy_deg)
        payload["rotate"] = {
            "z": _clean(z),
            "x": _clean(x),
            "y": _clean(y),
            "order": "ZXY",
        }
    if action.translate_local_fraction is not None:
        x, y, z = np.asarray(action.translate_local_fraction)
        payload["translate"] = {"x": _clean(x), "y": _clean(y), "z": _clean(z)}
    if action.scale_fraction is not None:
        x, y, z = np.asarray(action.scale_fraction)
        payload["scale"] = {"x": _clean(x), "y": _clean(y), "z": _clean(z)}
    if not payload:
        raise ValueError("update_pose must contain at least one update")
    return f"<update_pose>{json.dumps(payload, separators=(',', ':'))}</update_pose>"
