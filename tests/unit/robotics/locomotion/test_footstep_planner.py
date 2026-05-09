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


class TestFootstepPlanner:
    """Tests for FootstepPlanner class."""

    def test_create_planner(self) -> None:
        """Test creating footstep planner."""
        params = GaitParameters()
        planner = FootstepPlanner(params)

        assert planner.parameters is params

    def test_plan_to_goal_straight(self) -> None:
        """Test planning straight path to goal."""
        params = GaitParameters(step_length=0.3)
        planner = FootstepPlanner(params)

        plan = planner.plan_to_goal(
            start=np.zeros(3),
            goal=np.array([1.0, 0.0, 0.0]),
        )

        assert plan.n_steps > 0

        # Steps should alternate
        for i, step in enumerate(plan.footsteps):
            expected_foot = "left" if i % 2 == 0 else "right"
            assert step.foot == expected_foot

        # Final step should be near goal
        final_step = plan.footsteps[-1]
        assert abs(final_step.position[0] - 1.0) < 0.2

    def test_plan_to_goal_already_there(self) -> None:
        """Test planning when already at goal."""
        params = GaitParameters()
        planner = FootstepPlanner(params)

        plan = planner.plan_to_goal(
            start=np.zeros(3),
            goal=np.zeros(3),
        )

        assert plan.n_steps == 0

    def test_plan_from_velocity(self) -> None:
        """Test planning from velocity command."""
        params = GaitParameters(step_duration=0.5)
        planner = FootstepPlanner(params)

        plan = planner.plan_from_velocity(
            current_position=np.zeros(3),
            current_yaw=0.0,
            velocity_command=np.array([0.5, 0.0, 0.0]),
            n_steps=4,
        )

        assert plan.n_steps == 4

        # Steps should progress forward
        for i in range(1, len(plan.footsteps)):
            assert (
                plan.footsteps[i].position[0] > plan.footsteps[i - 1].position[0] - 0.1
            )

    def test_plan_from_velocity_with_rotation(self) -> None:
        """Test planning with rotational velocity."""
        params = GaitParameters(step_duration=0.5)
        planner = FootstepPlanner(params)

        plan = planner.plan_from_velocity(
            current_position=np.zeros(3),
            current_yaw=0.0,
            velocity_command=np.array([0.0, 0.0, 0.5]),  # Rotation only
            n_steps=4,
        )

        assert plan.n_steps == 4

        # Final orientation should be different
        final_yaw = plan.footsteps[-1].yaw
        assert abs(final_yaw) > 0.1

    def test_plan_in_place_turn(self) -> None:
        """Test in-place turn planning."""
        params = GaitParameters()
        planner = FootstepPlanner(params)

        plan = planner.plan_in_place_turn(
            current_position=np.zeros(3),
            current_yaw=0.0,
            target_yaw=np.pi / 2,  # 90 degrees
        )

        assert plan.n_steps > 0

        # Position should stay near origin
        for step in plan.footsteps:
            assert np.linalg.norm(step.position[:2]) < 0.3

    def test_plan_respects_step_limits(self) -> None:
        """Test that planner respects step length limits."""
        params = GaitParameters(step_length=0.3)
        planner = FootstepPlanner(params, max_step_length=0.4)

        plan = planner.plan_from_velocity(
            current_position=np.zeros(3),
            current_yaw=0.0,
            velocity_command=np.array([2.0, 0.0, 0.0]),  # Very fast
            n_steps=4,
        )

        # Check step lengths are limited
        for i in range(1, len(plan.footsteps)):
            step_length = np.linalg.norm(
                plan.footsteps[i].position[:2] - plan.footsteps[i - 1].position[:2]
            )
            # Allow some margin for lateral offset
            assert step_length < 0.8
