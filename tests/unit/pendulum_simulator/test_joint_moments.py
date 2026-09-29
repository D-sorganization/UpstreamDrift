"""Tests for src.shared.python.pendulum_simulator.joint_moments (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.joint_moments import (
    cross_2d,
    double_pendulum_moments,
    moment_of_force,
    total_moment_at_joint,
    triple_pendulum_moments,
)


class TestCross2d:
    def test_orthogonal_unit_vectors(self) -> None:
        r = np.array([1.0, 0.0])
        f = np.array([0.0, 1.0])
        assert cross_2d(r, f) == pytest.approx(1.0)

    def test_parallel_vectors_zero(self) -> None:
        r = np.array([2.0, 0.0])
        f = np.array([3.0, 0.0])
        assert cross_2d(r, f) == pytest.approx(0.0)

    def test_negative_cross_product(self) -> None:
        r = np.array([0.0, 1.0])
        f = np.array([1.0, 0.0])
        assert cross_2d(r, f) == pytest.approx(-1.0)

    def test_zero_force(self) -> None:
        r = np.array([1.0, 2.0])
        f = np.array([0.0, 0.0])
        assert cross_2d(r, f) == pytest.approx(0.0)

    def test_joint_moments_returns_float(self) -> None:
        r = np.array([1.0, 0.0])
        f = np.array([0.0, 1.0])
        assert isinstance(cross_2d(r, f), float)

    def test_joint_moments_wrong_shape_raises(self) -> None:
        r = np.array([1.0, 0.0, 0.0])  # shape (3,)
        f = np.array([0.0, 1.0, 0.0])
        with pytest.raises(AssertionError):
            cross_2d(r, f)


class TestMomentOfForce:
    def test_horizontal_force_at_unit_lever(self) -> None:
        joint = np.array([0.0, 0.0])
        com = np.array([0.5, 0.0])
        force = np.array([0.0, 10.0])  # upward
        # r = (0.5, 0), F = (0, 10) → M = 0.5*10 - 0*0 = 5.0
        assert moment_of_force(joint, com, force) == pytest.approx(5.0)

    def test_zero_force_zero_moment(self) -> None:
        joint = np.array([0.0, 0.0])
        com = np.array([1.0, 0.0])
        force = np.array([0.0, 0.0])
        assert moment_of_force(joint, com, force) == pytest.approx(0.0)

    def test_com_at_joint_zero_moment(self) -> None:
        joint = np.array([1.0, 1.0])
        com = np.array([1.0, 1.0])  # same as joint
        force = np.array([5.0, 10.0])
        assert moment_of_force(joint, com, force) == pytest.approx(0.0)


class TestTotalMomentAtJoint:
    def test_applied_plus_force_moment(self) -> None:
        joint = np.array([0.0, 0.0])
        com = np.array([0.5, 0.0])
        force = np.array([0.0, 10.0])
        applied = 3.0
        # moment_of_force = 5.0, total = 3.0 + 5.0 = 8.0
        assert total_moment_at_joint(applied, joint, com, force) == pytest.approx(8.0)

    def test_zero_applied_torque(self) -> None:
        joint = np.array([0.0, 0.0])
        com = np.array([0.5, 0.0])
        force = np.array([0.0, 10.0])
        assert total_moment_at_joint(0.0, joint, com, force) == pytest.approx(5.0)

    def test_cancellation(self) -> None:
        joint = np.array([0.0, 0.0])
        com = np.array([0.5, 0.0])
        force = np.array([0.0, 10.0])
        applied = -5.0
        assert total_moment_at_joint(applied, joint, com, force) == pytest.approx(0.0)


class TestDoublePendulumMoments:
    def _make_positions(self) -> dict:
        return {
            "shoulder": np.array([0.0, 0.0]),
            "wrist": np.array([0.0, -0.5]),
            "tip": np.array([0.0, -1.0]),
        }

    def _make_forces(self) -> dict:
        return {
            "shoulder": np.array([1.0, 0.0]),
            "wrist": np.array([0.5, 0.0]),
        }

    def test_joint_moments_returns_dict(self) -> None:
        result = double_pendulum_moments(
            self._make_positions(), self._make_forces(), (10.0, 5.0), None
        )
        assert isinstance(result, dict)

    def test_has_shoulder_and_wrist_keys(self) -> None:
        result = double_pendulum_moments(
            self._make_positions(), self._make_forces(), (10.0, 5.0), None
        )
        assert "shoulder_applied_torque" in result
        assert "wrist_applied_torque" in result

    def test_applied_torques_stored(self) -> None:
        result = double_pendulum_moments(
            self._make_positions(), self._make_forces(), (10.0, 5.0), None
        )
        assert result["shoulder_applied_torque"] == pytest.approx(10.0)
        assert result["wrist_applied_torque"] == pytest.approx(5.0)

    def test_has_six_keys(self) -> None:
        result = double_pendulum_moments(
            self._make_positions(), self._make_forces(), (10.0, 5.0), None
        )
        assert len(result) == 6


class TestTriplePendulumMoments:
    def _make_positions(self) -> dict:
        return {
            "shoulder": np.array([0.0, 0.0]),
            "elbow": np.array([0.0, -0.3]),
            "wrist": np.array([0.0, -0.7]),
            "tip": np.array([0.0, -1.0]),
        }

    def _make_forces(self) -> dict:
        return {
            "shoulder": np.array([1.0, 0.0]),
            "elbow": np.array([0.5, 0.0]),
            "wrist": np.array([0.2, 0.0]),
        }

    def test_joint_moments_returns_dict(self) -> None:
        result = triple_pendulum_moments(
            self._make_positions(), self._make_forces(), (5.0, 3.0, 2.0), None
        )
        assert isinstance(result, dict)

    def test_has_nine_keys(self) -> None:
        result = triple_pendulum_moments(
            self._make_positions(), self._make_forces(), (5.0, 3.0, 2.0), None
        )
        assert len(result) == 9

    def test_applied_torques_stored(self) -> None:
        result = triple_pendulum_moments(
            self._make_positions(), self._make_forces(), (5.0, 3.0, 2.0), None
        )
        assert result["shoulder_applied_torque"] == pytest.approx(5.0)
        assert result["elbow_applied_torque"] == pytest.approx(3.0)
        assert result["wrist_applied_torque"] == pytest.approx(2.0)
