"""Comprehensive tests for deployment.realtime.controller."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.deployment.realtime.controller import (
    RealTimeController,
    RobotConfig,
    TimingStatistics,
)
from src.deployment.realtime.state import (
    ControlCommand,
    ControlMode,
    RobotState,
)


def _config(n: int = 3) -> RobotConfig:
    return RobotConfig(name="bot", n_joints=n)


def _callback_zero(state: RobotState) -> ControlCommand:
    return ControlCommand.torque_command(
        state.timestamp, np.zeros(len(state.joint_positions))
    )


class TestRobotConfig:
    def test_default_joint_names(self) -> None:
        c = RobotConfig(name="bot", n_joints=3)
        assert c.joint_names == ["joint_0", "joint_1", "joint_2"]

    def test_explicit_joint_names_kept(self) -> None:
        c = RobotConfig(name="bot", n_joints=2, joint_names=["a", "b"])
        assert c.joint_names == ["a", "b"]


class TestTimingStatisticsDefaults:
    def test_defaults(self) -> None:
        s = TimingStatistics()
        assert s.total_cycles == 0
        assert s.min_cycle_time == float("inf")


class TestConnect:
    def test_connect_simulation(self) -> None:
        c = RealTimeController(communication_type="simulation")
        assert c.connect(_config()) is True
        assert c.is_connected

    def test_connect_loopback(self) -> None:
        c = RealTimeController(communication_type="loopback")
        assert c.connect(_config()) is True

    def test_connect_ros2_fails(self) -> None:
        c = RealTimeController(communication_type="ros2")
        assert c.connect(_config()) is False
        assert not c.is_connected

    def test_connect_udp_fails(self) -> None:
        c = RealTimeController(communication_type="udp")
        assert c.connect(_config()) is False

    def test_connect_ethercat_fails(self) -> None:
        c = RealTimeController(communication_type="ethercat")
        assert c.connect(_config()) is False

    def test_disconnect_clears(self) -> None:
        c = RealTimeController()
        c.connect(_config())
        c.disconnect()
        assert not c.is_connected
        assert c._config is None


class TestStartStop:
    def test_start_without_connect_raises(self) -> None:
        c = RealTimeController()
        c.set_control_callback(_callback_zero)
        with pytest.raises(RuntimeError, match="connect"):
            c.start()

    def test_start_without_callback_raises(self) -> None:
        c = RealTimeController()
        c.connect(_config())
        with pytest.raises(RuntimeError, match="callback"):
            c.start()

    def test_start_twice_noop(self) -> None:
        c = RealTimeController(control_frequency=100.0)
        c.connect(_config())
        c.set_control_callback(_callback_zero)
        c.start()
        try:
            c.start()  # second start should be no-op
            assert c.is_running
        finally:
            c.stop()

    def test_full_cycle_simulation(self) -> None:
        c = RealTimeController(control_frequency=200.0)
        c.connect(_config())
        c.set_control_callback(_callback_zero)
        c.start()
        time.sleep(0.05)
        c.stop()
        assert not c.is_running
        stats = c.get_timing_stats()
        assert stats.total_cycles > 0

    def test_get_timing_stats_no_data(self) -> None:
        c = RealTimeController()
        stats = c.get_timing_stats()
        assert stats.total_cycles == 0

    def test_disconnect_while_running(self) -> None:
        c = RealTimeController(control_frequency=200.0)
        c.connect(_config())
        c.set_control_callback(_callback_zero)
        c.start()
        time.sleep(0.02)
        c.disconnect()
        assert not c.is_connected


class TestLoopback:
    def test_loopback_torque(self) -> None:
        c = RealTimeController(control_frequency=100.0, communication_type="loopback")
        c.connect(_config(n=3))
        c._sim_state = (np.zeros(3), np.zeros(3))
        cmd = ControlCommand.torque_command(0.0, np.ones(3))
        c._send_command(cmd)
        q, qd = c._sim_state
        assert qd[0] > 0

    def test_loopback_position(self) -> None:
        c = RealTimeController(communication_type="loopback")
        c.connect(_config(n=3))
        c._sim_state = (np.zeros(3), np.ones(3))
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.POSITION,
            position_targets=np.array([1.0, 2.0, 3.0]),
        )
        c._send_command(cmd)
        q, qd = c._sim_state
        np.testing.assert_array_equal(q, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(qd, np.zeros(3))

    def test_loopback_velocity(self) -> None:
        c = RealTimeController(control_frequency=100.0, communication_type="loopback")
        c.connect(_config(n=3))
        c._sim_state = (np.zeros(3), np.zeros(3))
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.VELOCITY,
            velocity_targets=np.array([1.0, 0.0, 0.0]),
        )
        c._send_command(cmd)
        q, qd = c._sim_state
        assert qd[0] == 1.0

    def test_loopback_impedance(self) -> None:
        c = RealTimeController(control_frequency=100.0, communication_type="loopback")
        c.connect(_config(n=3))
        c._sim_state = (np.zeros(3), np.zeros(3))
        cmd = ControlCommand.impedance_command(
            0.0,
            np.ones(3),
            np.ones(3) * 10,
            np.ones(3),
            feedforward=np.zeros(3),
        )
        c._send_command(cmd)
        q, qd = c._sim_state
        assert qd[0] > 0

    def test_loopback_init_when_none(self) -> None:
        c = RealTimeController(communication_type="loopback")
        c.connect(_config(n=3))
        c._sim_state = None
        cmd = ControlCommand.torque_command(0.0, np.zeros(3))
        c._send_command(cmd)
        assert c._sim_state is not None

    def test_loopback_read_state(self) -> None:
        c = RealTimeController(communication_type="loopback")
        c.connect(_config(n=3))
        s = c._read_state()
        assert s.joint_positions.shape == (3,)

    def test_simulation_read_state(self) -> None:
        c = RealTimeController(communication_type="simulation")
        c.connect(_config(n=4))
        s = c._read_state()
        assert s.joint_positions.shape == (4,)


class TestWaitForState:
    def test_wait_for_state_timeout(self) -> None:
        c = RealTimeController()
        s = c.wait_for_state(timeout=0.01)
        assert s is None

    def test_get_last_state_none(self) -> None:
        c = RealTimeController()
        assert c.get_last_state() is None
        assert c.get_last_command() is None

    def test_wait_for_state_returns_when_set(self) -> None:
        c = RealTimeController(control_frequency=200.0, communication_type="loopback")
        c.connect(_config(n=3))
        c.set_control_callback(_callback_zero)
        c.start()
        try:
            s = c.wait_for_state(timeout=0.5)
            assert s is not None
        finally:
            c.stop()
