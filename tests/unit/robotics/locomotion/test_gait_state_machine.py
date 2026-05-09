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


class TestGaitStateMachine:
    """Tests for GaitStateMachine class."""

    def test_create_state_machine(self) -> None:
        """Test creating gait state machine."""
        gait = GaitStateMachine()

        assert gait.state.gait_type == GaitType.WALK
        assert not gait.is_walking

    def test_start_walking(self) -> None:
        """Test starting to walk."""
        gait = GaitStateMachine()
        gait.start_walking()

        assert gait.is_walking
        assert gait.state.phase == GaitPhase.DOUBLE_SUPPORT

    def test_stop_walking(self) -> None:
        """Test stopping walking."""
        gait = GaitStateMachine()
        gait.start_walking()
        gait.stop_walking()

        assert not gait.is_walking
        assert gait.state.gait_type == GaitType.STAND

    def test_emergency_stop(self) -> None:
        """Test emergency stop."""
        params = GaitParameters(step_duration=0.5)
        gait = GaitStateMachine(params)
        gait.start_walking()

        # Advance into swing phase
        gait.update(0.2)  # Past double support

        gait.emergency_stop()

        assert not gait.is_walking
        assert gait.state.phase == GaitPhase.DOUBLE_SUPPORT

    def test_emergency_stop_resets_stance_foot(self) -> None:
        """E-stop must reset stance_foot to 'both' (#2503)."""
        params = GaitParameters(step_duration=0.5)
        gait = GaitStateMachine(params)
        gait.start_walking()
        gait.update(0.2)  # Advance to single-support (stance_foot != 'both')

        gait.emergency_stop()

        assert gait.state.stance_foot == "both"

    def test_emergency_stop_resets_next_stance_foot(self) -> None:
        """E-stop must reset next_stance_foot to 'both' (#2503)."""
        params = GaitParameters(step_duration=0.5)
        gait = GaitStateMachine(params)
        gait.start_walking()
        gait.update(0.2)

        gait.emergency_stop()

        assert gait.state.next_stance_foot == "both"

    def test_emergency_stop_full_state_consistency(self) -> None:
        """After E-stop all stance fields must reflect double-support standing (#2503)."""
        params = GaitParameters(step_duration=0.5)
        gait = GaitStateMachine(params)
        gait.start_walking()
        gait.update(0.3)

        gait.emergency_stop()

        state = gait.state
        assert not state.is_walking
        assert state.gait_type == GaitType.STAND
        assert state.phase == GaitPhase.DOUBLE_SUPPORT
        assert state.stance_foot == "both"
        assert state.next_stance_foot == "both"

    def test_update_advances_time(self) -> None:
        """Test update advances phase time."""
        gait = GaitStateMachine()
        gait.start_walking()

        gait.update(0.05)

        assert gait.state.phase_time == 0.05
        assert gait.state.cycle_time == 0.05

    def test_phase_transition(self) -> None:
        """Test phase transitions during walking."""
        params = GaitParameters(
            step_duration=0.5,
            double_support_ratio=0.2,
        )
        gait = GaitStateMachine(params)
        gait.start_walking()

        # Start in double support
        assert gait.state.phase == GaitPhase.DOUBLE_SUPPORT

        # Advance past double support duration (0.1s)
        gait.update(0.15)

        # Should be in swing phase
        assert gait.state.phase in (GaitPhase.LEFT_SWING, GaitPhase.RIGHT_SWING)

    def test_step_count_increments(self) -> None:
        """Test step count increments after each step."""
        params = GaitParameters(
            step_duration=0.5,
            double_support_ratio=0.2,
        )
        gait = GaitStateMachine(params)
        gait.start_walking()

        initial_steps = gait.state.step_count

        # Complete two full phases (double support + swing + double support + swing)
        # Double support: 0.1s, Swing: 0.4s, need at least one complete cycle
        gait.update(1.1)  # More than two complete steps

        assert gait.state.step_count > initial_steps

    def test_phase_progress(self) -> None:
        """Test phase progress calculation."""
        params = GaitParameters(
            step_duration=0.5,
            double_support_ratio=0.2,
        )
        gait = GaitStateMachine(params)
        gait.start_walking()

        # At start, progress should be 0
        assert gait.phase_progress == 0.0

        # At half of double support (0.05s)
        gait.update(0.05)
        assert_allclose(gait.phase_progress, 0.5, atol=0.01)

    def test_callback_registration(self) -> None:
        """Test callback registration and invocation."""
        gait = GaitStateMachine()
        callback_invoked = [False]

        def on_gait_change(state: GaitState, event: GaitEvent) -> None:
            callback_invoked[0] = True

        gait.register_callback("gait_change", on_gait_change)
        gait.start_walking()

        assert callback_invoked[0]

    def test_foot_trajectory_phase(self) -> None:
        """Test foot trajectory phase calculation."""
        params = GaitParameters(
            step_duration=0.5,
            double_support_ratio=0.2,
        )
        gait = GaitStateMachine(params)

        # When standing, trajectory phase should be 1.0
        assert gait.get_foot_trajectory_phase("left") == 1.0
        assert gait.get_foot_trajectory_phase("right") == 1.0
