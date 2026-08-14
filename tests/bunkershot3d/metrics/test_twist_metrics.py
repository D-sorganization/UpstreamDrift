"""Tests for head twist metrics (issue #8614).

Moment about the shaft axis and about the CG — the reason bounce and relief exist.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.metrics.twist import (
    compute_twist_metrics,
)


class TestMomentAboutShaftAxis:
    """The shaft axis moment causes the club to twist in the golfer's hands."""

    def test_shaft_axis_moment_from_torque_projection(self) -> None:
        """M_shaft = M . shaft_axis_unit."""
        # Shaft axis typically along y (pointing up toward golfer's hands)
        shaft_axis = np.array([0.0, 1.0, 0.0])

        t = np.array([0.0, 0.005, 0.010])
        torques = np.array(
            [
                [1.0, 2.0, 0.5],  # M_shaft = 2.0 N.m
                [1.5, 3.0, 1.0],  # M_shaft = 3.0 N.m <- peak
                [0.5, 1.5, 0.3],  # M_shaft = 1.5 N.m
            ]
        )

        metrics = compute_twist_metrics(t, torques, shaft_axis)

        assert metrics.peak_shaft_moment_nm == pytest.approx(3.0, rel=1e-6)
        assert metrics.peak_shaft_moment_time_s == pytest.approx(0.005, rel=1e-6)

    def test_shaft_moment_sign_indicates_direction(self) -> None:
        """Positive moment opens the face, negative closes it."""
        shaft_axis = np.array([0.0, 1.0, 0.0])

        t = np.array([0.0, 0.005])
        torques = np.array(
            [
                [0.0, -2.0, 0.0],  # Closing torque
                [0.0, -3.0, 0.0],  # Stronger closing
            ]
        )

        metrics = compute_twist_metrics(t, torques, shaft_axis)

        # Peak is the maximum absolute value, sign preserved
        assert metrics.peak_shaft_moment_nm == pytest.approx(-3.0, rel=1e-6)

    def test_mean_shaft_moment(self) -> None:
        """Time-weighted mean moment about shaft."""
        shaft_axis = np.array([0.0, 1.0, 0.0])

        t = np.array([0.0, 0.005, 0.010])
        torques = np.array(
            [
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )

        metrics = compute_twist_metrics(t, torques, shaft_axis)

        assert metrics.mean_shaft_moment_nm == pytest.approx(2.0, rel=0.01)


class TestMomentAboutCG:
    """Moment about the clubhead's center of gravity."""

    def test_moment_about_cg_from_force_and_arm(self) -> None:
        """M_CG = r_contact x F + M_contact."""
        # CG position relative to contact point
        cg_position = np.array([0.02, 0.0, 0.01])  # 20mm forward, 10mm up

        t = np.array([0.0, 0.005])
        # Contact forces
        forces = np.array(
            [
                [0.0, 0.0, 500.0],  # Vertical force
                [0.0, 0.0, 1000.0],  # Larger vertical force
            ]
        )
        # Contact torques (torsional friction at contact)
        torques = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        # Contact points (where force is applied)
        contact_points = np.array(
            [
                [0.0, 0.0, -0.02],  # Leading edge
                [0.0, 0.0, -0.02],
            ]
        )

        metrics = compute_twist_metrics(
            t,
            torques,
            shaft_axis=np.array([0.0, 1.0, 0.0]),
            forces=forces,
            contact_points=contact_points,
            cg_position=cg_position,
        )

        # M = r x F where r = cg - contact
        # r = [0.02, 0, 0.01] - [0, 0, -0.02] = [0.02, 0, 0.03]
        # F = [0, 0, 1000]
        # M = [0*1000 - 0.03*0, 0.03*0 - 0.02*1000, 0.02*0 - 0*0]
        #   = [0, -20, 0] N.m
        assert abs(metrics.peak_cg_moment_nm) > 15  # Should be ~20 N.m


class TestTwistIntegral:
    """Integrated twist over the shot (total angular impulse)."""

    def test_twist_impulse_integral(self) -> None:
        """Impulse = integral of M dt."""
        shaft_axis = np.array([0.0, 1.0, 0.0])

        # Constant 2 N.m for 10 ms = 0.02 N.m.s
        t = np.array([0.0, 0.005, 0.010])
        torques = np.array(
            [
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )

        metrics = compute_twist_metrics(t, torques, shaft_axis)

        # Integral of 2 N.m over 0.010 s = 0.020 N.m.s
        assert metrics.shaft_impulse_nm_s == pytest.approx(0.020, rel=0.1)


class TestTwistDirection:
    """Which direction does the club want to twist?"""

    def test_opening_vs_closing_classification(self) -> None:
        """Determine if the net moment opens or closes the face."""
        shaft_axis = np.array([0.0, 1.0, 0.0])

        t = np.array([0.0, 0.005, 0.010])
        # Net positive moment about shaft = face opening
        torques = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )

        metrics = compute_twist_metrics(t, torques, shaft_axis)

        assert metrics.net_twist_direction == "opening"

    def test_closing_twist_direction(self) -> None:
        shaft_axis = np.array([0.0, 1.0, 0.0])

        t = np.array([0.0, 0.005, 0.010])
        torques = np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, -3.0, 0.0],
                [0.0, -2.0, 0.0],
            ]
        )

        metrics = compute_twist_metrics(t, torques, shaft_axis)

        assert metrics.net_twist_direction == "closing"


class TestTwistWithShaftOrientation:
    """Handle non-standard shaft axis orientations."""

    def test_angled_shaft_axis(self) -> None:
        """Shaft axis at typical lie angle (~60 deg from vertical)."""
        # Shaft points up and back
        lie_angle_rad = math.radians(60)
        shaft_axis = np.array(
            [
                0.0,
                math.sin(lie_angle_rad),
                math.cos(lie_angle_rad),
            ]
        )
        shaft_axis = shaft_axis / np.linalg.norm(shaft_axis)

        t = np.array([0.0, 0.005])
        torques = np.array(
            [
                [0.0, 2.0, 1.0],
                [0.0, 2.0, 1.0],
            ]
        )

        metrics = compute_twist_metrics(t, torques, shaft_axis)

        # Projection onto shaft axis
        m_dot_s = torques[0] @ shaft_axis
        assert metrics.peak_shaft_moment_nm == pytest.approx(m_dot_s, rel=0.01)
