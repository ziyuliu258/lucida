import numpy as np
import pytest

from lucida_mini.metrics import oriented_box_iou
from lucida_mini.pose import Pose


def box(position=(0, 0, 0), size=(2, 2, 2), rotation=None):
    return Pose(
        position=np.asarray(position, dtype=float),
        rotation=np.eye(3) if rotation is None else np.asarray(rotation, dtype=float),
        size=np.asarray(size, dtype=float),
    )


def test_oriented_box_iou_identity_and_disjoint():
    assert oriented_box_iou(box(), box()) == pytest.approx(1.0)
    assert oriented_box_iou(box(), box(position=(3, 0, 0))) == 0.0


def test_oriented_box_iou_half_overlap():
    # Two 2x2x2 boxes overlap in a 1x2x2 slab: IoU = 4 / (8+8-4).
    assert oriented_box_iou(box(), box(position=(1, 0, 0))) == pytest.approx(1 / 3)


def test_oriented_box_iou_uses_orientation():
    angle = np.deg2rad(45)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    value = oriented_box_iou(box(size=(2, 1, 1)), box(size=(2, 1, 1), rotation=rotation))
    assert 0.0 < value < 1.0


def test_oriented_box_iou_handles_thin_boxes_at_large_world_coordinates():
    """Regression for the CA-1M book-cover geometry.

    The boxes differ by sub-millimetre translation and sub-percent scale, so
    their volumetric overlap must remain high even though their world Z
    coordinate is roughly 294 metres.
    """
    target = Pose(
        position=np.array([1.0910856724, 50.8667297363, 293.8139343262]),
        rotation=np.array([
            [-0.4758739471, 0.8795135021, 0.0],
            [-0.8795135021, -0.4758739471, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        size=np.array([0.3037335575, 0.2175250202, 0.0113124680]),
    )
    prediction = Pose(
        position=np.array([1.0913500398, 50.8673862348, 293.8139494150]),
        rotation=np.array([
            [-0.4759384619, 0.8794785924, -0.0000024907],
            [-0.8794785910, -0.4759384610, 0.0000575660],
            [0.0000494426, 0.0000295884, 0.9999999983],
        ]),
        size=np.array([0.3048984503, 0.2183729150, 0.0112590360]),
    )
    assert oriented_box_iou(prediction, target) > 0.95
