"""Tests for real-time control module."""

from __future__ import annotations

import time

import numpy as np
import pytest


class TestRobotState:
    """Tests for RobotState dataclass."""

    def test_robot_state_creation(self) -> None:
        """Test creating a robot state."""
        from src.deployment.realtime import RobotState

        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            joint_torques=np.zeros(7),
        )

        assert state.n_joints == 7
        assert state.timestamp == 0.0

    def test_robot_state_vector(self) -> None:
        """Test getting state vector."""
        from src.deployment.realtime import RobotState

        positions = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float64)
        velocities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)

        state = RobotState(
            timestamp=0.0,
            joint_positions=positions,
            joint_velocities=velocities,
            joint_torques=np.zeros(7),
        )

        state_vector = state.get_state_vector()
        assert len(state_vector) == 14
        np.testing.assert_array_equal(state_vector[:7], positions)
        np.testing.assert_array_equal(state_vector[7:], velocities)


class TestControlCommand:
    """Tests for ControlCommand dataclass."""

    def test_position_command(self) -> None:
        """Test creating a position command."""
        from src.deployment.realtime import ControlCommand, ControlMode

        cmd = ControlCommand.position_command(
            timestamp=0.0,
            positions=np.zeros(7),
        )

        assert cmd.mode == ControlMode.POSITION
        assert cmd.position_targets is not None
        assert len(cmd.position_targets) == 7

    def test_torque_command(self) -> None:
        """Test creating a torque command."""
        from src.deployment.realtime import ControlCommand, ControlMode

        cmd = ControlCommand.torque_command(
            timestamp=0.0,
            torques=np.ones(7),
        )

        assert cmd.mode == ControlMode.TORQUE
        assert cmd.torque_commands is not None
        np.testing.assert_array_equal(cmd.torque_commands, np.ones(7))

    def test_impedance_command(self) -> None:
        """Test creating an impedance command."""
        from src.deployment.realtime import ControlCommand, ControlMode

        cmd = ControlCommand.impedance_command(
            timestamp=0.0,
            positions=np.zeros(7),
            stiffness=np.ones(7) * 100,
            damping=np.ones(7) * 10,
        )

        assert cmd.mode == ControlMode.IMPEDANCE
        assert cmd.stiffness is not None
        assert cmd.damping is not None

    def test_command_validation(self) -> None:
        """Test command validation."""
        from src.deployment.realtime import ControlCommand, ControlMode

        # Valid position command
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.POSITION,
            position_targets=np.zeros(7),
        )
        assert cmd.validate(7)

        # Invalid: missing position_targets
        cmd_invalid = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.POSITION,
        )
        with pytest.raises(ValueError):
            cmd_invalid.validate(7)


class TestRealTimeController:
    """Tests for RealTimeController."""

    def test_controller_creation(self) -> None:
        """Test creating a controller."""
        from src.deployment.realtime import RealTimeController

        controller = RealTimeController(
            control_frequency=1000.0,
            communication_type="simulation",
        )

        assert controller.control_frequency == 1000.0
        assert controller.dt == 0.001
        assert not controller.is_connected
        assert not controller.is_running

    def test_controller_connect(self) -> None:
        """Test connecting to simulated robot."""
        from src.deployment.realtime import RealTimeController, RobotConfig

        controller = RealTimeController(communication_type="simulation")
        config = RobotConfig(name="test_robot", n_joints=7)

        success = controller.connect(config)
        assert success
        assert controller.is_connected

        controller.disconnect()
        assert not controller.is_connected

    def test_controller_timing_stats(self) -> None:
        """Test timing statistics."""
        from src.deployment.realtime import (
            ControlCommand,
            ControlMode,
            RealTimeController,
            RobotConfig,
            RobotState,
        )

        controller = RealTimeController(
            control_frequency=100.0,  # Low frequency for test
            communication_type="simulation",
        )
        config = RobotConfig(name="test_robot", n_joints=7)
        controller.connect(config)

        def simple_callback(state: RobotState) -> ControlCommand:
            return ControlCommand(
                timestamp=state.timestamp,
                mode=ControlMode.TORQUE,
                torque_commands=np.zeros(7),
            )

        controller.set_control_callback(simple_callback)
        controller.start()

        # Run briefly
        time.sleep(0.05)

        controller.stop()

        stats = controller.get_timing_stats()
        assert stats.total_cycles > 0
        assert stats.mean_cycle_time > 0

        controller.disconnect()

    def test_loopback_physics(self) -> None:
        """Test LOOPBACK physics simulation."""
        from src.deployment.realtime import (
            ControlCommand,
            ControlMode,
            RealTimeController,
            RobotConfig,
        )

        controller = RealTimeController(
            control_frequency=100.0,
            communication_type="loopback",
        )
        config = RobotConfig(name="test_robot", n_joints=1)
        controller.connect(config)

        # Trigger initialization
        controller._read_state()

        # Test TORQUE mode
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.TORQUE,
            torque_commands=np.array([1.0]),
        )
        controller._send_command(cmd)

        q, qd = controller._sim_state  # type: ignore
        # After 1 step (dt=0.01) with tau=1: v=0.01, p=0.0001
        assert qd[0] > 0
        assert q[0] > 0

        # Test VELOCITY mode
        cmd_vel = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.VELOCITY,
            velocity_targets=np.array([2.0]),
        )
        controller._send_command(cmd_vel)

        q_new, qd_new = controller._sim_state  # type: ignore
        assert qd_new[0] == 2.0
        assert q_new[0] > q[0]

        # Test POSITION mode
        cmd_pos = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.POSITION,
            position_targets=np.array([10.0]),
        )
        controller._send_command(cmd_pos)

        q_pos, qd_pos = controller._sim_state  # type: ignore
        assert q_pos[0] == 10.0
        assert qd_pos[0] == 0.0


class TestControlLoopFailureEscalation:
    """Tests for consecutive-failure escalation (issue #6943)."""

    def test_loop_aborts_and_zeroes_torque_after_n_failures(self) -> None:
        """A persistently failing callback aborts the loop and zeroes torque."""
        from src.deployment.realtime import (
            RealTimeController,
            RobotConfig,
            RobotState,
        )

        controller = RealTimeController(
            control_frequency=200.0,
            communication_type="simulation",
            max_consecutive_failures=3,
        )
        controller.connect(RobotConfig(name="test_robot", n_joints=2))

        sent: list[object] = []
        controller._send_command = sent.append  # type: ignore[method-assign]

        def failing_callback(_state: RobotState) -> object:
            raise RuntimeError("simulated hardware fault")

        controller.set_control_callback(failing_callback)
        controller.start()

        # Wait for the loop to self-abort.
        deadline = time.perf_counter() + 2.0
        while controller.is_running and time.perf_counter() < deadline:
            time.sleep(0.005)

        assert not controller.is_running
        assert controller.aborted_on_failure
        # A zero-torque command was issued as the safety fallback.
        assert sent, "expected a zero-torque safety command"
        last = sent[-1]
        np.testing.assert_array_equal(
            last.torque_commands,  # type: ignore[attr-defined]
            np.zeros(2),
        )

    def test_transient_failures_do_not_abort(self) -> None:
        """An occasional failure that recovers must not abort the loop."""
        from src.deployment.realtime import (
            ControlCommand,
            ControlMode,
            RealTimeController,
            RobotConfig,
            RobotState,
        )

        controller = RealTimeController(
            control_frequency=200.0,
            communication_type="simulation",
            max_consecutive_failures=5,
        )
        controller.connect(RobotConfig(name="test_robot", n_joints=1))

        state = {"calls": 0}

        def flaky_callback(s: RobotState) -> ControlCommand:
            state["calls"] += 1
            if state["calls"] % 4 == 0:
                raise RuntimeError("transient")
            return ControlCommand(
                timestamp=s.timestamp,
                mode=ControlMode.TORQUE,
                torque_commands=np.zeros(1),
            )

        controller.set_control_callback(flaky_callback)
        controller.start()
        time.sleep(0.1)

        assert controller.is_running
        assert not controller.aborted_on_failure
        controller.stop()
        controller.disconnect()

    def test_invalid_max_consecutive_failures_rejected(self) -> None:
        """Precondition: max_consecutive_failures must be positive."""
        from src.deployment.realtime import RealTimeController

        with pytest.raises(ValueError):
            RealTimeController(max_consecutive_failures=0)


class TestControllerStopTimeout:
    """Tests for stop() join-timeout handling (issue #6944)."""

    def test_stop_raises_and_skips_zero_command_on_join_timeout(self) -> None:
        """If the thread won't stop, stop() raises and skips the zero command."""
        from src.deployment.realtime import RealTimeController, RobotConfig

        controller = RealTimeController(communication_type="simulation")
        controller.connect(RobotConfig(name="test_robot", n_joints=2))

        sent: list[object] = []
        controller._send_command = sent.append  # type: ignore[method-assign]

        class _StuckThread:
            def join(self, timeout: float | None = None) -> None:
                return None

            def is_alive(self) -> bool:
                return True

        controller._control_thread = _StuckThread()  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="did not stop"):
            controller.stop()

        # No zero command was sent because the loop is still alive.
        assert sent == []
        # The (still-alive) thread handle is retained, not cleared.
        assert controller._control_thread is not None

    def test_stop_sends_zero_command_when_thread_confirmed_stopped(self) -> None:
        """Normal stop confirms the join and commands zero torque once."""
        from src.deployment.realtime import (
            ControlCommand,
            ControlMode,
            RealTimeController,
            RobotConfig,
            RobotState,
        )

        controller = RealTimeController(
            control_frequency=200.0,
            communication_type="simulation",
        )
        controller.connect(RobotConfig(name="test_robot", n_joints=3))

        sent: list[object] = []
        controller._send_command = sent.append  # type: ignore[method-assign]

        def cb(s: RobotState) -> ControlCommand:
            return ControlCommand(
                timestamp=s.timestamp,
                mode=ControlMode.TORQUE,
                torque_commands=np.zeros(3),
            )

        controller.set_control_callback(cb)
        controller.start()
        time.sleep(0.03)
        controller.stop()

        assert not controller.is_running
        assert controller._control_thread is None
        assert sent, "expected a zero-torque command on clean stop"
        np.testing.assert_array_equal(
            sent[-1].torque_commands,  # type: ignore[attr-defined]
            np.zeros(3),
        )


class TestRobotConfig:
    """Tests for RobotConfig."""

    def test_realtime_config_defaults(self) -> None:
        """Test default configuration."""
        from src.deployment.realtime import RobotConfig

        config = RobotConfig(name="test", n_joints=7)

        assert config.name == "test"
        assert config.n_joints == 7
        assert len(config.joint_names) == 7
        assert config.joint_names[0] == "joint_0"
