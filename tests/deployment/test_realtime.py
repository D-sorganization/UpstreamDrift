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
            CommunicationType,
            ControlCommand,
            ControlMode,
            RealTimeController,
            RobotConfig,
        )

        controller = RealTimeController(
            control_frequency=100.0,
            communication_type="loopback",
        )
        config = RobotConfig(
            name="test_robot",
            n_joints=1,
            communication_type=CommunicationType.LOOPBACK,
        )
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

        # Access internal state from interface
        q, qd = controller._interface._sim_state  # type: ignore
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

        q_new, qd_new = controller._interface._sim_state  # type: ignore
        assert qd_new[0] == 2.0
        assert q_new[0] > q[0]

        # Test POSITION mode
        cmd_pos = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.POSITION,
            position_targets=np.array([10.0]),
        )
        controller._send_command(cmd_pos)

        q_pos, qd_pos = controller._interface._sim_state  # type: ignore
        assert q_pos[0] == 10.0
        assert qd_pos[0] == 0.0


class TestRobotConfig:
    """Tests for RobotConfig."""

    def test_config_defaults(self) -> None:
        """Test default configuration."""
        from src.deployment.realtime import RobotConfig

        config = RobotConfig(name="test", n_joints=7)

        assert config.name == "test"
        assert config.n_joints == 7
        assert len(config.joint_names) == 7
        assert config.joint_names[0] == "joint_0"
