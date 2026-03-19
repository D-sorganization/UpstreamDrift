import numpy as np
import pytest
from src.shared.python.screw_theory.kinematics import Twist, ScrewAxis, compute_screw_axis, compute_screw_endpoints

__all__ = ["test_pure_translation", "test_pure_rotation", "test_helical_motion", "test_endpoints_singular", "test_endpoints_general"]

def test_pure_translation() -> None:
    twist = Twist(
        angular=np.zeros(3),
        linear=np.array([1.0, 0.0, 0.0]),
        body_name="test",
        reference_point=np.array([0.0, 0.0, 0.0])
    )
    screw = compute_screw_axis(twist)
    assert screw.is_singular is True
    assert screw.pitch == float("inf")
    np.testing.assert_allclose(screw.axis_direction, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(screw.axis_point, twist.reference_point)

def test_pure_rotation() -> None:
    twist = Twist(
        angular=np.array([0.0, 0.0, 1.0]),
        linear=np.array([0.0, 0.0, 0.0]),
        body_name="test",
        reference_point=np.array([0.0, 0.0, 0.0])
    )
    screw = compute_screw_axis(twist)
    assert screw.is_singular is False
    assert screw.pitch == 0.0
    np.testing.assert_allclose(screw.axis_direction, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(screw.axis_point, np.array([0.0, 0.0, 0.0]))

def test_helical_motion() -> None:
    twist = Twist(
        angular=np.array([0.0, 0.0, 2.0]),
        linear=np.array([1.0, 0.0, 4.0]),
        body_name="test",
        reference_point=np.array([0.0, 0.0, 0.0])
    )
    screw = compute_screw_axis(twist)
    assert screw.is_singular is False
    assert screw.pitch == 2.0  # h = (w dot v) / w^2 = (2*4) / 4 = 2.0
    np.testing.assert_allclose(screw.axis_direction, np.array([0.0, 0.0, 1.0]))
    # axis point = r + (w x v) / w^2 = [0,0,0] + ([-2, 0, 0] / 4) = [-0.5, 0, 0]
    expected_q = np.array([0.0, 0.5, 0.0])  # wait, w = [0,0,2], v = [1,0,4]. w x v = [0*4-2*0, 2*1-0*4, 0*0-0*1] = [0, 2, 0]. q = [0, 0.5, 0].
    np.testing.assert_allclose(screw.axis_point, expected_q)

def test_endpoints_singular() -> None:
    twist = Twist(
        angular=np.zeros(3),
        linear=np.array([1.0, 0.0, 0.0]),
        body_name="test",
        reference_point=np.array([2.0, 3.0, 4.0])
    )
    screw = compute_screw_axis(twist)
    start, end = compute_screw_endpoints(screw, length=10.0)
    np.testing.assert_allclose(start, screw.axis_point)
    np.testing.assert_allclose(end, screw.axis_point + np.array([10.0, 0.0, 0.0]))

def test_endpoints_general() -> None:
    twist = Twist(
        angular=np.array([0.0, 0.0, 1.0]),
        linear=np.array([0.0, 0.0, 0.0]),
        body_name="test",
        reference_point=np.array([0.0, 0.0, 0.0])
    )
    screw = compute_screw_axis(twist)
    start, end = compute_screw_endpoints(screw, length=2.0)
    np.testing.assert_allclose(start, np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(end, np.array([0.0, 0.0, 1.0]))
