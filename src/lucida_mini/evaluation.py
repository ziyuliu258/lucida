"""Evaluation rules for overfitting the frozen expert trajectories."""
from __future__ import annotations


def rollout_protocol(max_steps: int = 24) -> dict:
    return {
        "name": "fixed_trajectory_error_replay",
        "version": 3,
        "max_steps": max_steps,
        "observation_views": "separate_native_raw_overlay_colored_pointcloud_plus_optional_local_axes",
        "ca1m_frame_schedule": "one_frame_per_turn_in_capture_order_then_hold_last",
        "prompt_version": "target-cue-v1",
        "add_sb_success_fraction": 0.05,
        "rotation_success_deg": 5.0,
    }


def iter_rollout_schedule(trajectory, max_steps: int):
    """Replay only masked errors; let the model use the full turn budget.

    Supervised expert answers are never supplied as the model's next action.
    Once the recorded trajectory ends, all remaining turns are model turns.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    for index in range(max_steps):
        turn = trajectory.turns[index] if index < len(trajectory.turns) else None
        injection = turn.action if turn is not None and not turn.supervise else None
        yield index, injection


def full_loop_success(row: dict) -> float:
    """Require the fixed target's orientation as well as geometry and stop."""
    return float(
        float(row["stopped"]) == 1.0
        and float(row["valid_action_rate"]) == 1.0
        and float(row["actions"]) > 0.0
        and float(row["add_sb_fraction"]) < 0.05
        and float(row["rotation_error_deg"]) < 5.0
    )
