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


class TestIntegration:
    """Integration tests for locomotion module."""

    def test_full_walking_cycle(self) -> None:
        """Test complete walking cycle."""
        # Setup
        params = GaitParameters(
            step_length=0.3,
            step_duration=0.5,
            double_support_ratio=0.2,
        )
        gait = GaitStateMachine(params)
        planner = FootstepPlanner(params)

        # Plan path
        plan = planner.plan_to_goal(
            start=np.zeros(3),
            goal=np.array([1.5, 0.0, 0.0]),
        )

        # Execute walking
        gait.start_walking()

        dt = 0.01
        max_time = plan.total_duration + 1.0
        time = 0.0

        while time < max_time and gait.is_walking:
            gait.update(dt)
            time += dt

            # Check state consistency
            state = gait.state
            assert state.phase_time >= 0
            assert state.cycle_time >= 0

    def test_zmp_during_walking(self) -> None:
        """Test ZMP remains valid during walking."""
        engine = MockHumanoidEngine()
        zmp_computer = ZMPComputer(engine)

        # Simulate various CoM states during walking
        test_states = [
            (np.array([0.0, 0.0, 0.9]), np.zeros(3)),  # Stationary
            (np.array([0.1, 0.0, 0.9]), np.array([0.5, 0.0, 0.0])),  # Forward
            (np.array([0.0, 0.05, 0.9]), np.array([0.0, 0.2, 0.0])),  # Lateral
        ]

        for com_pos, com_accel in test_states:
            result = zmp_computer.compute_zmp(
                com_position=com_pos,
                com_acceleration=com_accel,
            )

            # ZMP should be finite
            assert np.all(np.isfinite(result.zmp_position))
