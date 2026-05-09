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


class TestFootstepPlan:
    """Tests for FootstepPlan dataclass."""

    def test_create_empty_plan(self) -> None:
        """Test creating empty plan."""
        plan = FootstepPlan()

        assert len(plan) == 0
        assert plan.n_steps == 0

    def test_plan_iteration(self) -> None:
        """Test iterating over plan."""
        footsteps = [
            Footstep(np.zeros(3), np.array([1, 0, 0, 0]), "left", step_index=0),
            Footstep(
                np.array([0.3, 0, 0]), np.array([1, 0, 0, 0]), "right", step_index=1
            ),
        ]
        plan = FootstepPlan(footsteps=footsteps)

        steps = list(plan)
        assert len(steps) == 2

    def test_get_footsteps_for_foot(self) -> None:
        """Test filtering footsteps by foot."""
        footsteps = [
            Footstep(np.zeros(3), np.array([1, 0, 0, 0]), "left"),
            Footstep(np.zeros(3), np.array([1, 0, 0, 0]), "right"),
            Footstep(np.zeros(3), np.array([1, 0, 0, 0]), "left"),
        ]
        plan = FootstepPlan(footsteps=footsteps)

        left_steps = plan.get_footsteps_for_foot("left")
        assert len(left_steps) == 2
