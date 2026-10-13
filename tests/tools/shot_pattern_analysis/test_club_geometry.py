"""Independent rigid-rotation identities for the phase-two delivery model."""

import numpy as np
import pytest

from src.tools.shot_pattern_analysis.club_geometry import club_delivery

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("loft", [10.0, 34.0, 46.0])
@pytest.mark.parametrize("lean", [0.0, 5.0, 10.0])
def test_forward_lean_pitches_face_and_shaft(loft, lean):
    result = club_delivery(loft, 63.0, lean, 0.0, 0.0)
    assert result.loft_deg == pytest.approx(loft - lean)
    assert result.face_deg == pytest.approx(0.0)
    axis = np.asarray(result.shaft_axis)
    assert np.degrees(np.arctan2(axis[0], axis[2])) == pytest.approx(lean)
    assert np.linalg.norm(axis) == pytest.approx(1.0)


def test_rigid_rotations_preserve_face_shaft_angle_and_contact_basis():
    reference = club_delivery(34.0, 63.0, 0.0, 0.0, 0.0)
    for lean, yaw, twist in [(10, 3, 2), (5, -3, -2), (0, 2, 1)]:
        result = club_delivery(34.0, 63.0, lean, yaw, twist)
        assert np.dot(result.normal, result.shaft_axis) == pytest.approx(
            np.dot(reference.normal, reference.shaft_axis)
        )
        frame = np.column_stack((result.normal, result.heel_axis, result.up_axis))
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-14)
        assert np.linalg.det(frame) == pytest.approx(1.0)


def test_positive_shaft_twist_closes_and_delofts_and_yaw_is_equivariant():
    square = club_delivery(34, 63, 5, 0, 0)
    closed = club_delivery(34, 63, 5, 0, 1)
    yawed = club_delivery(34, 63, 5, 3, 1)
    assert closed.face_deg < square.face_deg
    assert closed.loft_deg < square.loft_deg
    assert yawed.face_deg == pytest.approx(closed.face_deg + 3)
    assert yawed.loft_deg == pytest.approx(closed.loft_deg)


@pytest.mark.parametrize(
    "args",
    [
        (float("nan"), 63, 0, 0, 0),
        (90, 63, 0, 0, 0),
        (34, 0, 0, 0, 0),
        (34, 63, 60, 0, 0),
        (34, 63, 0, 40, 0),
        (34, 63, 0, 0, 40),
    ],
)
def test_invalid_inputs_fail_before_rotation(args):
    with pytest.raises(ValueError):
        club_delivery(*args)
