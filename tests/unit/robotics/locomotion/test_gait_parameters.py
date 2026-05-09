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


class TestGaitParameters:
    """Tests for GaitParameters dataclass."""

    def test_default_parameters(self) -> None:
        """Test default parameter values."""
        params = GaitParameters()

        assert params.gait_type == GaitType.WALK
        assert params.step_length == 0.3
        assert params.step_duration == 0.5
        assert 0 <= params.double_support_ratio <= 1

    def test_locomotion_custom_parameters(self) -> None:
        """Test custom parameter values."""
        params = GaitParameters(
            step_length=0.4,
            step_duration=0.6,
            com_height=1.0,
        )

        assert params.step_length == 0.4
        assert params.step_duration == 0.6
        assert params.com_height == 1.0

    def test_locomotion_parameter_validation(self) -> None:
        """Test parameter validation."""
        with pytest.raises(ValueError, match="non-negative"):
            GaitParameters(step_length=-0.1)

        with pytest.raises(ValueError, match="positive"):
            GaitParameters(step_duration=0)

        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            GaitParameters(double_support_ratio=1.5)

    def test_swing_duration(self) -> None:
        """Test swing duration calculation."""
        params = GaitParameters(
            step_duration=0.5,
            double_support_ratio=0.2,
        )

        assert params.swing_duration == 0.4
        assert params.double_support_duration == 0.1

    def test_step_frequency(self) -> None:
        """Test step frequency calculation."""
        params = GaitParameters(step_duration=0.5)
        assert params.step_frequency == 2.0

    def test_create_walk_parameters(self) -> None:
        """Test walk parameter factory."""
        params = create_walk_parameters(step_length=0.35)

        assert params.gait_type == GaitType.WALK
        assert params.step_length == 0.35
        assert params.double_support_ratio > 0

    def test_create_run_parameters(self) -> None:
        """Test run parameter factory."""
        params = create_run_parameters()

        assert params.gait_type == GaitType.RUN
        assert params.double_support_ratio == 0.0  # No double support

    def test_create_stand_parameters(self) -> None:
        """Test stand parameter factory."""
        params = create_stand_parameters()

        assert params.gait_type == GaitType.STAND
        assert params.double_support_ratio == 1.0  # Always double support
        assert params.step_length == 0.0


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
