import itertools

import numpy as np
import pytest

from lucida_mini.actions import PermuteAxis, Stop, SwitchObs, UpdatePose, parse_action
from lucida_mini.expert import PrivilegedExpert, execute_action
from lucida_mini.metrics import add_sb, rotation_geodesic_deg, transform_normalized_points
from lucida_mini.pose import Pose, rotation_zxy
from lucida_mini.serialization import action_to_text
from lucida_mini.state import GizmoState, ObservationMode


def test_update_uses_new_rotation_and_old_size_for_translation() -> None:
    pose = Pose.identity(size=np.array([2.0, 3.0, 4.0]))
    updated = pose.update(
        rotate_zxy_deg=np.array([90.0, 0.0, 0.0]),
        translate_local_fraction=np.array([0.5, 0.0, 0.0]),
        scale_fraction=np.array([0.5, 0.0, 0.0]),
    )
    np.testing.assert_allclose(updated.position, [0.0, 1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(updated.size, [3.0, 3.0, 4.0])


def test_parser_accepts_paper_examples() -> None:
    assert isinstance(parse_action('<switch_obs>permute_axis</switch_obs>'), SwitchObs)
    assert isinstance(parse_action('<stop>{}</stop>'), Stop)
    assert isinstance(
        parse_action('<permute_axis>{"x":"-y","z":"x"}</permute_axis>'),
        PermuteAxis,
    )
    action = parse_action(
        '<update_pose>{"rotate":{"z":8.25,"x":-2.5,"y":3.1,"order":"ZXY"},'
        '"translate":{"x":0.1,"y":-0.04,"z":0.02},'
        '"scale":{"x":0.05,"y":-0.03,"z":0.02}}</update_pose>'
    )
    assert isinstance(action, UpdatePose)


@pytest.mark.parametrize(
    "text",
    [
        '<stop>{"unexpected":1}</stop>',
        '<update_pose>{}</update_pose>',
        '<update_pose>{"rotate":{"z":0,"x":0,"y":0,"order":"XYZ"}}</update_pose>',
        '<update_pose>{"rotate":1}</update_pose>',
        '<permute_axis>{"x":"x","z":"-x"}</permute_axis>',
        'prefix <stop>{}</stop>',
    ],
)
def test_parser_rejects_invalid_actions(text: str) -> None:
    with pytest.raises(ValueError):
        parse_action(text)


def test_exactly_24_right_handed_axis_permutations() -> None:
    names = ("x", "-x", "y", "-y", "z", "-z")
    valid = []
    for x, z in itertools.product(names, repeat=2):
        try:
            valid.append(PermuteAxis(x, z).matrix())
        except ValueError:
            pass
    assert len(valid) == 24
    assert all(np.isclose(np.linalg.det(matrix), 1.0) for matrix in valid)


def test_identity_metrics() -> None:
    points = np.array(
        [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
    )
    pose = Pose.identity()
    transformed = transform_normalized_points(points, pose)
    assert add_sb(transformed, transformed) == pytest.approx(0.0)
    assert rotation_geodesic_deg(pose, pose) == pytest.approx(0.0)


def test_action_round_trip() -> None:
    action = UpdatePose(
        rotate_zxy_deg=np.array([8.25, -2.5, 3.1]),
        translate_local_fraction=np.array([0.1, -0.04, 0.02]),
        scale_fraction=np.array([0.05, -0.03, 0.02]),
    )
    parsed = parse_action(action_to_text(action))
    assert isinstance(parsed, UpdatePose)
    np.testing.assert_allclose(parsed.rotate_zxy_deg, action.rotate_zxy_deg)
    np.testing.assert_allclose(parsed.translate_local_fraction, action.translate_local_fraction)
    np.testing.assert_allclose(parsed.scale_fraction, action.scale_fraction)


def test_expert_reaches_target_and_stops() -> None:
    target = Pose(
        position=np.array([0.2, -0.1, 0.4]),
        rotation=rotation_zxy(np.array([80.0, -15.0, 25.0])),
        size=np.array([0.7, 1.2, 0.9]),
    )
    current = Pose.identity(size=np.array([1.0, 1.0, 1.0]))
    expert = PrivilegedExpert(target)
    actions = []
    for _ in range(12):
        action = expert.act(current)
        actions.append(action)
        if isinstance(action, Stop):
            break
        current = execute_action(GizmoState(current), action).pose
    assert isinstance(actions[-1], Stop)
    assert rotation_geodesic_deg(current, target) <= 0.5
    assert np.linalg.norm((target.position - current.position) / target.size) <= 0.01
    assert np.max(np.abs(np.log(target.size / current.size))) <= 0.01


def test_switch_obs_changes_only_observation_mode() -> None:
    pose = Pose(
        position=np.array([0.2, -0.1, 0.4]),
        rotation=rotation_zxy(np.array([25.0, -10.0, 3.0])),
        size=np.array([0.7, 1.2, 0.9]),
    )
    before = GizmoState(pose)
    after = execute_action(before, parse_action("<switch_obs>permute_axis</switch_obs>"))
    assert after.observation_mode is ObservationMode.SIX_AXIS
    np.testing.assert_array_equal(after.pose.position, before.pose.position)
    np.testing.assert_array_equal(after.pose.rotation, before.pose.rotation)
    np.testing.assert_array_equal(after.pose.size, before.pose.size)
    back = execute_action(after, parse_action("<switch_obs>permute_axis</switch_obs>"))
    assert back.observation_mode is ObservationMode.SCENE
    np.testing.assert_array_equal(back.pose.position, before.pose.position)


def test_expert_stops_when_residual_quantizes_to_zero() -> None:
    target = Pose(
        position=np.array([0.004, 0.004, 0.0]),
        rotation=np.eye(3),
        size=np.ones(3),
    )
    expert = PrivilegedExpert(target)
    assert isinstance(expert.act(Pose.identity()), Stop)
