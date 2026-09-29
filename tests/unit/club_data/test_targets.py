"""Tests for src.shared.python.club_data.targets (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.club_data.targets import TargetTrajectory


def _make_trajectory(n: int = 10) -> TargetTrajectory:
    times = np.linspace(0.0, 1.0, n)
    positions = np.column_stack(
        [
            np.linspace(0.0, 1.0, n),
            np.zeros(n),
            np.linspace(0.0, 0.5, n),
        ]
    )
    return TargetTrajectory(name="test", time_series=times, positions=positions)


class TestTargetTrajectory:
    def test_targets_construction(self) -> None:
        traj = _make_trajectory()
        assert isinstance(traj, TargetTrajectory)

    def test_targets_name_stored(self) -> None:
        traj = _make_trajectory()
        assert traj.name == "test"

    def test_duration(self) -> None:
        traj = _make_trajectory()
        assert traj.duration == pytest.approx(1.0)

    def test_num_frames(self) -> None:
        traj = _make_trajectory(10)
        assert traj.num_frames == 10

    def test_default_color(self) -> None:
        traj = _make_trajectory()
        assert traj.color == pytest.approx((0.2, 0.8, 0.2))

    def test_default_opacity(self) -> None:
        traj = _make_trajectory()
        assert traj.opacity == pytest.approx(0.7)

    def test_velocities_default_none(self) -> None:
        traj = _make_trajectory()
        assert traj.velocities is None

    def test_get_position_at_start(self) -> None:
        traj = _make_trajectory(10)
        pos = traj.get_position_at_time(0.0)
        np.testing.assert_allclose(pos, traj.positions[0])

    def test_get_position_at_end(self) -> None:
        traj = _make_trajectory(10)
        pos = traj.get_position_at_time(1.0)
        np.testing.assert_allclose(pos, traj.positions[-1])

    def test_get_position_before_start_returns_first(self) -> None:
        traj = _make_trajectory(10)
        pos = traj.get_position_at_time(-1.0)
        np.testing.assert_allclose(pos, traj.positions[0])

    def test_get_position_after_end_returns_last(self) -> None:
        traj = _make_trajectory(10)
        pos = traj.get_position_at_time(5.0)
        np.testing.assert_allclose(pos, traj.positions[-1])

    def test_get_position_midpoint_interpolated(self) -> None:
        # Positions go from (0, 0, 0) to (1, 0, 0.5) linearly
        traj = _make_trajectory(11)
        pos = traj.get_position_at_time(0.5)
        assert pos.shape == (3,)
        # x should be ~0.5 at midpoint
        assert 0.4 < pos[0] < 0.6

    def test_get_position_returns_ndarray(self) -> None:
        traj = _make_trajectory()
        pos = traj.get_position_at_time(0.5)
        assert isinstance(pos, np.ndarray)

    def test_get_velocity_without_velocities_returns_none(self) -> None:
        traj = _make_trajectory()
        vel = traj.get_velocity_at_time(0.5)
        assert vel is None

    def test_get_velocity_with_velocities(self) -> None:
        times = np.linspace(0.0, 1.0, 10)
        positions = np.zeros((10, 3))
        velocities = np.ones((10, 3)) * 2.0
        traj = TargetTrajectory(
            name="with_vel",
            time_series=times,
            positions=positions,
            velocities=velocities,
        )
        vel = traj.get_velocity_at_time(0.5)
        assert vel is not None
        assert vel.shape == (3,)

    def test_custom_phase_markers(self) -> None:
        traj = TargetTrajectory(
            name="phased",
            time_series=np.linspace(0, 1, 10),
            positions=np.zeros((10, 3)),
            address_idx=2,
            top_idx=5,
            impact_idx=8,
            finish_idx=9,
        )
        assert traj.address_idx == 2
        assert traj.impact_idx == 8
