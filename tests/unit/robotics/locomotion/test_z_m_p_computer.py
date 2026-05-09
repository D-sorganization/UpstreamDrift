"""Unit tests for locomotion module.

Tests cover:
    - Gait types and parameters
    - ZMP computation
    - Gait state machine
    - Footstep planning
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from src.robotics.locomotion.footstep_planner import (
    Footstep,
    FootstepPlan,
    FootstepPlanner,
)
from src.robotics.locomotion.gait_state_machine import (
    GaitEvent,
    GaitState,
    GaitStateMachine,
)
from src.robotics.locomotion.gait_types import (
    GaitParameters,
    GaitPhase,
    GaitType,
    SupportState,
    create_run_parameters,
    create_stand_parameters,
    create_walk_parameters,
)
from src.robotics.locomotion.zmp_computer import (
    ZMPComputer,
)


class MockHumanoidEngine:
    """Mock humanoid engine for ZMP tests."""

    def __init__(self) -> None:
        self._com = np.array([0.0, 0.0, 0.9])
        self._com_vel = np.zeros(3)
        self._mass = 70.0

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(10), np.zeros(10)

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        pass

    def compute_mass_matrix(self) -> np.ndarray:
        return np.eye(10)

    def compute_bias_forces(self) -> np.ndarray:
        return np.zeros(10)

    def compute_gravity_forces(self) -> np.ndarray:
        return np.zeros(10)

    def compute_jacobian(self, body_name: str) -> dict | None:
        return {"linear": np.zeros((3, 10)), "angular": np.zeros((3, 10))}

    def get_time(self) -> float:
        return 0.0

    def get_com_position(self) -> np.ndarray:
        return self._com.copy()

    def get_com_velocity(self) -> np.ndarray:
        return self._com_vel.copy()

    def get_total_mass(self) -> float:
        return self._mass


class TestZMPComputer:
    """Tests for ZMPComputer class."""

    def test_create_zmp_computer(self) -> None:
        """Test creating ZMP computer."""
        engine = MockHumanoidEngine()
        zmp = ZMPComputer(engine)

        assert zmp.ground_height == 0.0

    def test_compute_zmp_stationary(self) -> None:
        """Test ZMP computation for stationary robot."""
        engine = MockHumanoidEngine()
        zmp = ZMPComputer(engine)

        result = zmp.compute_zmp(
            com_position=np.array([0.0, 0.0, 0.9]),
            com_acceleration=np.zeros(3),
        )

        # Stationary robot: ZMP should be directly below CoM
        assert_allclose(result.zmp_position[:2], [0.0, 0.0], atol=1e-6)
        assert result.zmp_position[2] == 0.0

    def test_compute_zmp_accelerating(self) -> None:
        """Test ZMP computation for accelerating robot."""
        engine = MockHumanoidEngine()
        zmp = ZMPComputer(engine)

        # Accelerating forward shifts ZMP backward
        result = zmp.compute_zmp(
            com_position=np.array([0.0, 0.0, 0.9]),
            com_acceleration=np.array([1.0, 0.0, 0.0]),  # Forward accel
        )

        # ZMP should be behind CoM
        assert result.zmp_position[0] < 0

    def test_compute_zmp_validity(self) -> None:
        """Test ZMP validity checking."""
        engine = MockHumanoidEngine()
        zmp = ZMPComputer(engine)

        # ZMP at origin with small support polygon
        support = np.array(
            [
                [-0.1, -0.1],
                [0.1, -0.1],
                [0.1, 0.1],
                [-0.1, 0.1],
            ]
        )

        result = zmp.compute_zmp(
            com_position=np.array([0.0, 0.0, 0.9]),
            com_acceleration=np.zeros(3),
            support_polygon=support,
        )

        assert result.is_valid
        assert result.support_margin > 0

    def test_compute_zmp_outside_support(self) -> None:
        """Test ZMP outside support polygon."""
        engine = MockHumanoidEngine()
        zmp = ZMPComputer(engine)

        # Large acceleration to push ZMP outside
        support = np.array(
            [
                [-0.05, -0.05],
                [0.05, -0.05],
                [0.05, 0.05],
                [-0.05, 0.05],
            ]
        )

        result = zmp.compute_zmp(
            com_position=np.array([0.0, 0.0, 0.9]),
            com_acceleration=np.array([5.0, 0.0, 0.0]),  # Large accel
            support_polygon=support,
        )

        assert not result.is_valid
        assert result.support_margin < 0

    def test_compute_capture_point(self) -> None:
        """Test capture point computation."""
        engine = MockHumanoidEngine()
        zmp = ZMPComputer(engine)

        # Moving forward, capture point should be ahead
        capture = zmp.compute_capture_point(
            com_position=np.array([0.0, 0.0, 0.9]),
            com_velocity=np.array([0.5, 0.0, 0.0]),
        )

        assert capture[0] > 0  # Ahead of CoM

    def test_compute_dcm(self) -> None:
        """Test DCM computation (equivalent to capture point)."""
        engine = MockHumanoidEngine()
        zmp = ZMPComputer(engine)

        com_pos = np.array([0.0, 0.0, 0.9])
        com_vel = np.array([0.3, 0.1, 0.0])

        capture = zmp.compute_capture_point(com_pos, com_vel)
        dcm = zmp.compute_dcm(com_pos, com_vel)

        assert_allclose(capture, dcm)

    def test_stability_margin(self) -> None:
        """Test stability margin computation."""
        engine = MockHumanoidEngine()
        zmp = ZMPComputer(engine)

        support = np.array(
            [
                [-0.1, -0.1],
                [0.1, -0.1],
                [0.1, 0.1],
                [-0.1, 0.1],
            ]
        )

        # Point at center
        margin = zmp.compute_stability_margin(
            np.array([0.0, 0.0]),
            support,
        )
        assert margin > 0
        assert_allclose(margin, 0.1, atol=1e-6)

        # Point at edge
        margin = zmp.compute_stability_margin(
            np.array([0.1, 0.0]),
            support,
        )
        assert_allclose(margin, 0.0, atol=1e-6)
