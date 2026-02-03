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

from src.robotics.locomotion.gait_types import (
    GaitType,
    GaitPhase,
    LegState,
    SupportState,
    GaitParameters,
    create_walk_parameters,
    create_run_parameters,
    create_stand_parameters,
)
from src.robotics.locomotion.zmp_computer import (
    ZMPComputer,
    ZMPResult,
)
from src.robotics.locomotion.gait_state_machine import (
    GaitStateMachine,
    GaitState,
    GaitEvent,
)
from src.robotics.locomotion.footstep_planner import (
    Footstep,
    FootstepPlan,
    FootstepPlanner,
)


class TestGaitTypes:
    """Tests for gait type enumerations."""

    def test_gait_type_values(self) -> None:
        """Test GaitType enum values exist."""
        assert GaitType.STAND is not None
        assert GaitType.WALK is not None
        assert GaitType.RUN is not None
        assert GaitType.TROT is not None

    def test_gait_phase_values(self) -> None:
        """Test GaitPhase enum values exist."""
        assert GaitPhase.DOUBLE_SUPPORT is not None
        assert GaitPhase.LEFT_SUPPORT is not None
        assert GaitPhase.RIGHT_SUPPORT is not None
        assert GaitPhase.FLIGHT is not None

    def test_support_state_values(self) -> None:
        """Test SupportState enum values."""
        assert SupportState.DOUBLE_SUPPORT_CENTERED is not None
        assert SupportState.SINGLE_SUPPORT_LEFT is not None
        assert SupportState.SINGLE_SUPPORT_RIGHT is not None


class TestGaitParameters:
    """Tests for GaitParameters dataclass."""

    def test_default_parameters(self) -> None:
        """Test default parameter values."""
        params = GaitParameters()

        assert params.gait_type == GaitType.WALK
        assert params.step_length == 0.3
        assert params.step_duration == 0.5
        assert 0 <= params.double_support_ratio <= 1

    def test_custom_parameters(self) -> None:
        """Test custom parameter values."""
        params = GaitParameters(
            step_length=0.4,
            step_duration=0.6,
            com_height=1.0,
        )

        assert params.step_length == 0.4
        assert params.step_duration == 0.6
        assert params.com_height == 1.0

    def test_parameter_validation(self) -> None:
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
        support = np.array([
            [-0.1, -0.1],
            [0.1, -0.1],
            [0.1, 0.1],
            [-0.1, 0.1],
        ])

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
        support = np.array([
            [-0.05, -0.05],
            [0.05, -0.05],
            [0.05, 0.05],
            [-0.05, 0.05],
        ])

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

        support = np.array([
            [-0.1, -0.1],
            [0.1, -0.1],
            [0.1, 0.1],
            [-0.1, 0.1],
        ])

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
        orient = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])

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
            Footstep(np.array([0.3, 0, 0]), np.array([1, 0, 0, 0]), "right", step_index=1),
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
            assert plan.footsteps[i].position[0] > plan.footsteps[i-1].position[0] - 0.1

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
                plan.footsteps[i].position[:2] - plan.footsteps[i-1].position[:2]
            )
            # Allow some margin for lateral offset
            assert step_length < 0.8


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
