import numpy as np
import pytest

from lucida_mini.metrics import evaluate_pose
from lucida_mini.pose import Pose


def test_identity_pose_has_perfect_metrics():
    rng = np.random.default_rng(3)
    points = rng.uniform(-0.5, 0.5, size=(100, 3))
    values = evaluate_pose(points, Pose.identity(), Pose.identity())
    assert values["add_sb_m"] == pytest.approx(0.0)
    assert values["iou_3d"] == pytest.approx(1.0)
    assert values["rotation_error_deg"] == pytest.approx(0.0)
    assert values["add_sb_at_0.01"] == 1.0
    assert values["rotation_at_5deg"] == 1.0
