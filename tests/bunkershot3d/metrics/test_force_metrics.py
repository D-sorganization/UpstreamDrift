"""Tests for force metrics (issue #8614).

Peak and mean head deceleration; peak resultant force and moment.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.metrics.force import (
    compute_force_metrics,
)


class TestPeakForce:
    """Peak resultant force on the head."""

    def test_peak_force_magnitude(self) -> None:
        """Peak force is the maximum |F| over the trajectory."""
        t = np.array([0.0, 0.005, 0.010, 0.015])
        forces = np.array(
            [
                [0.0, 0.0, 0.0],
                [300.0, 0.0, 400.0],  # |F| = 500 N
                [600.0, 0.0, 800.0],  # |F| = 1000 N <- peak
                [200.0, 0.0, 300.0],  # |F| = 361 N
            ]
        )
        torques = np.zeros((4, 3))

        metrics = compute_force_metrics(t, forces, torques, head_mass_kg=0.30)

        assert metrics.peak_force_n == pytest.approx(1000.0, rel=1e-6)
        assert metrics.peak_force_time_s == pytest.approx(0.010, rel=1e-6)

    def test_mean_force(self) -> None:
        """Mean force is time-weighted average |F|."""
        t = np.array([0.0, 0.005, 0.010])
        forces = np.array(
            [
                [300.0, 0.0, 400.0],  # |F| = 500 N
                [300.0, 0.0, 400.0],  # |F| = 500 N
                [300.0, 0.0, 400.0],  # |F| = 500 N
            ]
        )
        torques = np.zeros((3, 3))

        metrics = compute_force_metrics(t, forces, torques, head_mass_kg=0.30)

        assert metrics.mean_force_n == pytest.approx(500.0, rel=0.01)


class TestPeakMoment:
    """Peak resultant moment on the head."""

    def test_peak_moment_magnitude(self) -> None:
        """Peak moment is the maximum |M| over the trajectory."""
        t = np.array([0.0, 0.005, 0.010])
        forces = np.zeros((3, 3))
        torques = np.array(
            [
                [1.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],  # |M| = 5 N.m <- peak
                [1.0, 1.0, 1.0],  # |M| = sqrt(3) N.m
            ]
        )

        metrics = compute_force_metrics(t, forces, torques, head_mass_kg=0.30)

        assert metrics.peak_moment_nm == pytest.approx(5.0, rel=1e-6)
        assert metrics.peak_moment_time_s == pytest.approx(0.005, rel=1e-6)


class TestHeadDeceleration:
    """Peak and mean deceleration of the clubhead."""

    def test_peak_deceleration_from_velocity_change(self) -> None:
        """a = dv/dt, peak is maximum."""
        # Velocity drops from 25 to 10 m/s over 10 ms
        # Average decel = 1500 m/s^2 = ~153 g
        t = np.array([0.0, 0.005, 0.010])
        velocities = np.array(
            [
                [25.0, 0.0, 0.0],
                [17.5, 0.0, 0.0],  # -1500 m/s^2
                [10.0, 0.0, 0.0],  # -1500 m/s^2
            ]
        )
        forces = np.zeros((3, 3))
        torques = np.zeros((3, 3))

        metrics = compute_force_metrics(
            t, forces, torques, head_mass_kg=0.30, velocities=velocities
        )

        # Deceleration ~ 1500 m/s^2 in x direction
        assert metrics.peak_deceleration_m_s2 == pytest.approx(1500.0, rel=0.1)

    def test_mean_deceleration(self) -> None:
        """Mean deceleration is average over contact duration."""
        t = np.array([0.0, 0.005, 0.010])
        velocities = np.array(
            [
                [25.0, 0.0, 0.0],
                [17.5, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )
        forces = np.zeros((3, 3))
        torques = np.zeros((3, 3))

        metrics = compute_force_metrics(
            t, forces, torques, head_mass_kg=0.30, velocities=velocities
        )

        # (25-10) / 0.010 = 1500 m/s^2
        assert metrics.mean_deceleration_m_s2 == pytest.approx(1500.0, rel=0.1)

    def test_deceleration_from_force_fallback(self) -> None:
        """When velocities unavailable, compute a = F/m."""
        t = np.array([0.0, 0.005, 0.010])
        forces = np.array(
            [
                [450.0, 0.0, 0.0],  # a = 1500 m/s^2
                [450.0, 0.0, 0.0],
                [450.0, 0.0, 0.0],
            ]
        )
        torques = np.zeros((3, 3))
        head_mass_kg = 0.30

        metrics = compute_force_metrics(t, forces, torques, head_mass_kg=head_mass_kg)

        # F/m = 450/0.30 = 1500 m/s^2
        assert metrics.peak_deceleration_m_s2 == pytest.approx(1500.0, rel=0.01)


class TestForceComponents:
    """Individual force components for analysis."""

    def test_force_components_available(self) -> None:
        """Access to x, y, z components of peak force."""
        t = np.array([0.0, 0.005, 0.010])
        forces = np.array(
            [
                [100.0, 50.0, 200.0],
                [300.0, 100.0, 400.0],  # Peak magnitude
                [150.0, 75.0, 250.0],
            ]
        )
        torques = np.zeros((3, 3))

        metrics = compute_force_metrics(t, forces, torques, head_mass_kg=0.30)

        assert metrics.peak_force_components[0] == pytest.approx(300.0, rel=1e-6)
        assert metrics.peak_force_components[1] == pytest.approx(100.0, rel=1e-6)
        assert metrics.peak_force_components[2] == pytest.approx(400.0, rel=1e-6)


class TestContactDuration:
    """Time the club is in contact with sand."""

    def test_contact_duration_from_nonzero_force(self) -> None:
        """Contact duration is time where |F| > threshold."""
        t = np.array([0.0, 0.002, 0.004, 0.006, 0.008, 0.010])
        forces = np.array(
            [
                [0.0, 0.0, 0.0],  # No contact
                [50.0, 0.0, 0.0],  # Contact starts
                [200.0, 0.0, 0.0],  # Contact
                [300.0, 0.0, 0.0],  # Contact
                [100.0, 0.0, 0.0],  # Contact
                [0.0, 0.0, 0.0],  # Contact ends
            ]
        )
        torques = np.zeros((6, 3))

        metrics = compute_force_metrics(
            t, forces, torques, head_mass_kg=0.30, force_threshold_n=10.0
        )

        # Contact includes transition intervals (OR of adjacent points)
        # From t=0 to t=0.002 (one side in contact), t=0.002 to t=0.010
        # Total = 0.01 s (includes entry/exit intervals)
        assert metrics.contact_duration_s == pytest.approx(0.01, rel=0.2)
