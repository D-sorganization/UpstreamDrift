"""Tests for trajectory metrics (issue #8614).

Each test uses a synthetic trace with a hand-computed expected value.
"""

from __future__ import annotations


import numpy as np
import pytest

from bunkershot3d.metrics.trajectory import (
    compute_trajectory_metrics,
)


class TestDigVsSkid:
    """Net vertical force balance determines whether the club digs or skids."""

    def test_purely_vertical_force_is_dig(self) -> None:
        """Downward net force means the club is digging."""
        # Synthetic trace: club moving forward (x+), force mainly upward (z+)
        # But head accelerates downward => net force is down => digging
        t = np.array([0.0, 0.005, 0.010])
        positions = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, -0.01], [0.10, 0.0, -0.02]])
        forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 500.0], [0.0, 0.0, 800.0]])
        head_mass_kg = 0.30

        metrics = compute_trajectory_metrics(t, positions, forces, head_mass_kg)

        # Club moves down 20mm in 10ms while moving forward 100mm
        # Vertical velocity = -2 m/s, horizontal velocity = 10 m/s
        # This is a dig: z decreases monotonically
        assert metrics.is_digging
        assert not metrics.is_skidding

    def test_purely_horizontal_motion_is_skid(self) -> None:
        """Horizontal motion without depth change is skidding."""
        t = np.array([0.0, 0.005, 0.010])
        positions = np.array(
            [[0.0, 0.0, -0.01], [0.05, 0.0, -0.01], [0.10, 0.0, -0.01]]
        )
        forces = np.array([[0.0, 0.0, 300.0], [0.0, 0.0, 300.0], [0.0, 0.0, 300.0]])
        head_mass_kg = 0.30

        metrics = compute_trajectory_metrics(t, positions, forces, head_mass_kg)

        # Club stays at constant depth => skidding
        assert metrics.is_skidding
        assert not metrics.is_digging

    def test_dig_skid_index_ranges(self) -> None:
        """dig_skid_index: -1 is pure dig, +1 is pure skid, 0 is balanced."""
        # Balanced: equal dig and skid phases
        t = np.array([0.0, 0.005, 0.010, 0.015, 0.020])
        positions = np.array(
            [
                [0.00, 0.0, 0.00],
                [0.02, 0.0, -0.01],  # Digging
                [0.04, 0.0, -0.02],  # Digging
                [0.06, 0.0, -0.02],  # Skidding
                [0.08, 0.0, -0.02],  # Skidding
            ]
        )
        forces = np.zeros((5, 3))
        head_mass_kg = 0.30

        metrics = compute_trajectory_metrics(t, positions, forces, head_mass_kg)

        # Index between -1 and +1; this trace has 2 dig intervals, 1 skid interval
        # so slightly positive (more skid time in submerged portion)
        assert -1.0 <= metrics.dig_skid_index <= 1.0


class TestDepthProfile:
    """Entry point, maximum depth, exit point, divot dimensions."""

    def test_entry_point_is_first_submergence(self) -> None:
        """Entry point is where z first goes below 0 (surface)."""
        t = np.array([0.0, 0.005, 0.010, 0.015])
        positions = np.array(
            [
                [0.0, 0.0, 0.01],  # Above surface
                [0.02, 0.0, 0.0],  # At surface
                [0.04, 0.0, -0.02],  # Below surface (entry)
                [0.06, 0.0, -0.03],
            ]
        )
        forces = np.zeros((4, 3))

        metrics = compute_trajectory_metrics(t, positions, forces, 0.30)

        # Entry is interpolated between t=0.005 (z=0) and t=0.010 (z=-0.02)
        assert metrics.divot.entry_x_m == pytest.approx(0.02, rel=0.1)
        assert metrics.divot.entry_time_s == pytest.approx(0.005, rel=0.1)

    def test_maximum_depth_is_lowest_z(self) -> None:
        """Maximum depth is the lowest z coordinate."""
        t = np.array([0.0, 0.005, 0.010, 0.015, 0.020])
        positions = np.array(
            [
                [0.00, 0.0, 0.0],
                [0.02, 0.0, -0.01],
                [0.04, 0.0, -0.03],  # Maximum depth
                [0.06, 0.0, -0.02],
                [0.08, 0.0, 0.0],
            ]
        )
        forces = np.zeros((5, 3))

        metrics = compute_trajectory_metrics(t, positions, forces, 0.30)

        assert metrics.divot.max_depth_m == pytest.approx(0.03, abs=1e-6)
        assert metrics.divot.max_depth_x_m == pytest.approx(0.04, abs=1e-6)

    def test_exit_point_is_last_emergence(self) -> None:
        """Exit point is where z returns to 0."""
        t = np.array([0.0, 0.005, 0.010, 0.015, 0.020])
        positions = np.array(
            [
                [0.00, 0.0, 0.0],
                [0.02, 0.0, -0.02],
                [0.04, 0.0, -0.03],
                [0.06, 0.0, -0.01],
                [0.08, 0.0, 0.01],  # Exit point interpolated
            ]
        )
        forces = np.zeros((5, 3))

        metrics = compute_trajectory_metrics(t, positions, forces, 0.30)

        # Exit interpolated between z=-0.01 and z=0.01
        # Should be around x=0.07, t=0.0175
        assert 0.06 < metrics.divot.exit_x_m < 0.08
        assert 0.015 < metrics.divot.exit_time_s < 0.020

    def test_divot_length_is_entry_to_exit_distance(self) -> None:
        """Divot length is horizontal distance from entry to exit."""
        t = np.linspace(0, 0.020, 5)
        positions = np.array(
            [
                [0.00, 0.0, 0.01],
                [0.02, 0.0, -0.01],  # Entry ~ x=0.01
                [0.04, 0.0, -0.02],
                [0.06, 0.0, -0.01],
                [0.08, 0.0, 0.01],  # Exit ~ x=0.07
            ]
        )
        forces = np.zeros((5, 3))

        metrics = compute_trajectory_metrics(t, positions, forces, 0.30)

        # Entry at ~x=0.01, exit at ~x=0.07 => length ~0.06m
        assert 0.04 < metrics.divot.length_m < 0.08


class TestDepthVsDistanceTrace:
    """The depth trace is the raw data for understanding club-sand interaction."""

    def test_depth_trace_has_same_length_as_input(self) -> None:
        t = np.linspace(0, 0.01, 10)
        positions = np.column_stack(
            [np.linspace(0, 0.1, 10), np.zeros(10), np.linspace(0, -0.02, 10)]
        )
        forces = np.zeros((10, 3))

        metrics = compute_trajectory_metrics(t, positions, forces, 0.30)

        assert len(metrics.depth_trace_m) == 10
        assert len(metrics.distance_trace_m) == 10

    def test_depth_trace_is_negative_z(self) -> None:
        """Depth is positive downward, z is positive upward."""
        t = np.array([0.0, 0.005, 0.010])
        positions = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, -0.02], [0.10, 0.0, -0.03]])
        forces = np.zeros((3, 3))

        metrics = compute_trajectory_metrics(t, positions, forces, 0.30)

        # Depth = -z for submerged points
        np.testing.assert_array_almost_equal(metrics.depth_trace_m, [0.0, 0.02, 0.03])


class TestDivotVolumeMass:
    """Divot volume and mass estimates from the swept path."""

    def test_divot_volume_scales_with_depth_and_width(self) -> None:
        """V ~ depth * length * width approximation."""
        t = np.linspace(0, 0.02, 5)
        # 80mm long divot, 30mm max depth
        positions = np.array(
            [
                [0.00, 0.0, 0.0],
                [0.02, 0.0, -0.02],
                [0.04, 0.0, -0.03],
                [0.06, 0.0, -0.02],
                [0.08, 0.0, 0.0],
            ]
        )
        forces = np.zeros((5, 3))
        sole_width_m = 0.015  # 15mm

        metrics = compute_trajectory_metrics(
            t, positions, forces, 0.30, sole_width_m=sole_width_m
        )

        # Volume should be nonzero and positive
        assert metrics.divot.volume_m3 > 0
        # Rough upper bound: 0.08 * 0.03 * 0.015 = 3.6e-5 m^3
        assert metrics.divot.volume_m3 < 5e-5

    def test_divot_mass_uses_bulk_density(self) -> None:
        """Mass = volume * bulk_density."""
        t = np.linspace(0, 0.02, 5)
        positions = np.array(
            [
                [0.00, 0.0, 0.0],
                [0.02, 0.0, -0.02],
                [0.04, 0.0, -0.03],
                [0.06, 0.0, -0.02],
                [0.08, 0.0, 0.0],
            ]
        )
        forces = np.zeros((5, 3))
        bulk_density = 1550  # kg/m^3

        metrics = compute_trajectory_metrics(
            t,
            positions,
            forces,
            0.30,
            sole_width_m=0.015,
            bulk_density_kg_m3=bulk_density,
        )

        expected_mass = metrics.divot.volume_m3 * bulk_density
        assert metrics.divot.mass_kg == pytest.approx(expected_mass, rel=1e-6)


class TestEdgeCases:
    """Handle degenerate inputs gracefully."""

    def test_empty_trajectory_returns_zero_metrics(self) -> None:
        t = np.array([])
        positions = np.empty((0, 3))
        forces = np.empty((0, 3))

        metrics = compute_trajectory_metrics(t, positions, forces, 0.30)

        assert metrics.divot.max_depth_m == 0.0
        assert metrics.divot.length_m == 0.0

    def test_above_surface_trajectory_has_no_divot(self) -> None:
        """Club never touches sand."""
        t = np.array([0.0, 0.005, 0.010])
        positions = np.array([[0.0, 0.0, 0.02], [0.05, 0.0, 0.01], [0.10, 0.0, 0.02]])
        forces = np.zeros((3, 3))

        metrics = compute_trajectory_metrics(t, positions, forces, 0.30)

        assert metrics.divot.max_depth_m == 0.0
        assert not metrics.is_digging
        assert not metrics.is_skidding
