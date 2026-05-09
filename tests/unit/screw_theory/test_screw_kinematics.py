"""Tests for src.shared.python.screw_theory.kinematics (Issues #1949, #1744)."""

from __future__ import annotations

import math

import numpy as np
from src.shared.python.screw_theory.kinematics import (
    ScrewAxis,
    Twist,
    compute_screw_axis,
    compute_screw_endpoints,
)

# ---------------------------------------------------------------------------
# Twist dataclass
# ---------------------------------------------------------------------------


class TestTwist:
    def test_screw_kinematics_construct(self) -> None:
        t = Twist(
            angular=np.array([0.0, 0.0, 1.0]),
            linear=np.array([0.0, 0.0, 0.0]),
            body_name="test",
            reference_point=np.array([0.0, 0.0, 0.0]),
        )
        assert t.body_name == "test"

    def test_fields_accessible(self) -> None:
        ω = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        r = np.array([0.0, 0.0, 0.0])
        t = Twist(angular=ω, linear=v, body_name="b", reference_point=r)
        np.testing.assert_array_equal(t.angular, ω)
        np.testing.assert_array_equal(t.linear, v)


# ---------------------------------------------------------------------------
# ScrewAxis dataclass
# ---------------------------------------------------------------------------


class TestScrewAxis:
    def test_screw_kinematics_construct(self) -> None:
        sa = ScrewAxis(
            axis_direction=np.array([0.0, 0.0, 1.0]),
            axis_point=np.array([0.0, 0.0, 0.0]),
            pitch=0.0,
            angular_magnitude=1.0,
            linear_magnitude=0.0,
            is_singular=False,
        )
        assert sa.is_singular is False
        assert sa.pitch == 0.0


# ---------------------------------------------------------------------------
# compute_screw_axis — pure rotation
# ---------------------------------------------------------------------------


class TestComputeScrewAxisPureRotation:
    def _make_twist(self, ω, v, r=None) -> Twist:
        if r is None:
            r = np.zeros(3)
        return Twist(
            angular=np.array(ω),
            linear=np.array(v),
            body_name="b",
            reference_point=np.array(r),
        )

    def test_pure_rotation_z_axis(self) -> None:
        # Rotate about z-axis at origin: ω=[0,0,1], v=[0,0,0]
        t = self._make_twist([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        sa = compute_screw_axis(t)
        assert sa.is_singular is False
        np.testing.assert_allclose(sa.axis_direction, [0.0, 0.0, 1.0], atol=1e-10)

    def test_pure_rotation_pitch_zero(self) -> None:
        # Pure rotation (v perp to ω) → pitch = 0
        t = self._make_twist([0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
        sa = compute_screw_axis(t)
        assert abs(sa.pitch) < 1e-10

    def test_pure_rotation_angular_magnitude(self) -> None:
        t = self._make_twist([3.0, 4.0, 0.0], [0.0, 0.0, 0.0])
        sa = compute_screw_axis(t)
        assert abs(sa.angular_magnitude - 5.0) < 1e-10


# ---------------------------------------------------------------------------
# compute_screw_axis — pure translation (singular)
# ---------------------------------------------------------------------------


class TestComputeScrewAxisPureTranslation:
    def test_zero_angular_is_singular(self) -> None:
        t = Twist(
            angular=np.zeros(3),
            linear=np.array([1.0, 0.0, 0.0]),
            body_name="b",
            reference_point=np.zeros(3),
        )
        sa = compute_screw_axis(t)
        assert sa.is_singular is True

    def test_singular_pitch_is_inf(self) -> None:
        t = Twist(
            angular=np.zeros(3),
            linear=np.array([0.0, 1.0, 0.0]),
            body_name="b",
            reference_point=np.zeros(3),
        )
        sa = compute_screw_axis(t)
        assert math.isinf(sa.pitch)

    def test_no_motion_singular(self) -> None:
        t = Twist(
            angular=np.zeros(3),
            linear=np.zeros(3),
            body_name="b",
            reference_point=np.zeros(3),
        )
        sa = compute_screw_axis(t)
        assert sa.is_singular is True

    def test_singular_axis_direction_along_velocity(self) -> None:
        t = Twist(
            angular=np.zeros(3),
            linear=np.array([0.0, 0.0, 2.0]),
            body_name="b",
            reference_point=np.zeros(3),
        )
        sa = compute_screw_axis(t)
        np.testing.assert_allclose(sa.axis_direction, [0.0, 0.0, 1.0], atol=1e-10)


# ---------------------------------------------------------------------------
# compute_screw_endpoints
# ---------------------------------------------------------------------------


class TestComputeScrewEndpoints:
    def _simple_screw(self, singular: bool = False) -> ScrewAxis:
        return ScrewAxis(
            axis_direction=np.array([1.0, 0.0, 0.0]),
            axis_point=np.array([0.0, 0.0, 0.0]),
            pitch=0.0,
            angular_magnitude=1.0,
            linear_magnitude=0.0,
            is_singular=singular,
        )

    def test_screw_kinematics_returns_two_arrays(self) -> None:
        start, end = compute_screw_endpoints(self._simple_screw())
        assert start.shape == (3,)
        assert end.shape == (3,)

    def test_segment_length(self) -> None:
        start, end = compute_screw_endpoints(self._simple_screw(), length=1.0)
        length = float(np.linalg.norm(end - start))
        assert abs(length - 1.0) < 1e-10

    def test_singular_endpoint_offset(self) -> None:
        screw = self._simple_screw(singular=True)
        start, end = compute_screw_endpoints(screw, length=0.5)
        # start == axis_point for singular case
        np.testing.assert_allclose(start, screw.axis_point, atol=1e-10)
