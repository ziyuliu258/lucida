import json
from types import SimpleNamespace

import pytest

from lucida_mini.evaluation import full_loop_success, iter_rollout_schedule, rollout_protocol
from scripts.evaluate_checkpoints_parallel import current_closed_protocol


def test_fixed_errors_are_replayed_but_expert_stop_does_not_shorten_model_budget():
    error = '<update_pose>{"translate":{"x":0.01,"y":0,"z":0}}</update_pose>'
    trajectory = SimpleNamespace(turns=[
        SimpleNamespace(action=error, supervise=False),
        SimpleNamespace(action="expert correction", supervise=True),
        SimpleNamespace(action="<stop>{}</stop>", supervise=True),
    ])
    schedule = list(iter_rollout_schedule(trajectory, 12))
    assert len(schedule) == 12
    assert schedule[0] == (0, error)
    assert all(action is None for _, action in schedule[1:])
    assert list(iter_rollout_schedule(trajectory, 1)) == [(0, error)]


@pytest.mark.parametrize("field,value", [
    ("rotation_error_deg", 5.0),
    ("add_sb_fraction", 0.05),
    ("stopped", 0.0),
    ("valid_action_rate", 0.5),
    ("actions", 0),
    ("rotation_error_deg", float("nan")),
])
def test_fixed_target_success_rejects_bad_pose_or_protocol(field, value):
    row = dict(stopped=1, valid_action_rate=1, actions=3,
               add_sb_fraction=0.01, rotation_error_deg=0.1)
    assert full_loop_success(row) == 1.0
    row[field] = value
    assert full_loop_success(row) == 0.0


def test_previous_completed_results_are_not_reused_after_protocol_fix(tmp_path):
    path = tmp_path / "closed_loop.json"
    path.write_text(json.dumps({"summary": {}}))
    assert not current_closed_protocol(tmp_path)
    path.write_text(json.dumps({"protocol": rollout_protocol()}))
    assert current_closed_protocol(tmp_path)
    path.write_text(json.dumps({"protocol": rollout_protocol(7)}))
    assert not current_closed_protocol(tmp_path)
