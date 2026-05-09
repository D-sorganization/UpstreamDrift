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


class TestFootstep:
    """Tests for Footstep dataclass."""

    def test_create_footstep(self) -> None:
        """Test creating a footstep."""
        pos = np.array([0.3, 0.1, 0.0])
        orient = np.array([1.0, 0.0, 0.0, 0.0])

        step = Footstep(
            position=pos,
            orientation=orient,
            foot="left",
        )

        assert_array_equal(step.position, pos)
        assert step.foot == "left"

    def test_footstep_validation(self) -> None:
        """Test footstep validation."""
        with pytest.raises(ValueError, match="[Ff]oot"):
            Footstep(
                position=np.zeros(3),
                orientation=np.array([1, 0, 0, 0]),
                foot="middle",  # Invalid
            )

        with pytest.raises(ValueError, match="Position"):
            Footstep(
                position=np.zeros(2),  # Wrong shape
                orientation=np.array([1, 0, 0, 0]),
                foot="left",
            )

    def test_footstep_yaw(self) -> None:
        """Test yaw extraction from orientation."""
        # 90 degree rotation around z
        yaw = np.pi / 2
        orient = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

        step = Footstep(
            position=np.zeros(3),
            orientation=orient,
            foot="right",
        )

        assert_allclose(step.yaw, yaw, atol=1e-6)

    def test_footstep_pose_matrix(self) -> None:
        """Test pose matrix generation."""
        step = Footstep(
            position=np.array([1.0, 2.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity
            foot="left",
        )

        T = step.get_pose_matrix()

        assert T.shape == (4, 4)
        assert_allclose(T[:3, 3], [1.0, 2.0, 0.0])
        assert_allclose(T[:3, :3], np.eye(3), atol=1e-6)
