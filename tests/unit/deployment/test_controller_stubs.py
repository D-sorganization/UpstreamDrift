"""Tests for real-time controller stub detection (issue #2450).

Hardware-protocol connect paths (_connect_ros2, _connect_udp, _connect_ethercat)
must NOT silently claim success.  _read_state() and _send_command() for hardware
protocols must raise RuntimeError rather than returning fabricated zero values or
dropping commands silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.deployment.realtime import (
    ControlCommand,
    ControlMode,
    RobotConfig,
)
from src.deployment.realtime.controller import CommunicationType, RealTimeController


def _make_controller(comm: CommunicationType) -> RealTimeController:
    return RealTimeController(
        communication_type=comm.value,
        control_frequency=100.0,
    )


def _make_config(n_joints: int = 7) -> RobotConfig:
    return RobotConfig(name="test_robot", n_joints=n_joints)


class TestHardwareConnectStubs:
    """connect() must not set _is_connected=True for unimplemented hardware protocols."""

    def test_ros2_connect_returns_false(self) -> None:
        """ROS2 connect path is unimplemented; connect() must return False."""
        ctrl = _make_controller(CommunicationType.ROS2)
        result = ctrl.connect(_make_config())
        assert result is False, "Unimplemented ROS2 connect must return False"

    def test_ros2_is_not_connected_after_connect(self) -> None:
        """_is_connected must stay False after failed ROS2 connect."""
        ctrl = _make_controller(CommunicationType.ROS2)
        ctrl.connect(_make_config())
        assert ctrl._is_connected is False

    def test_udp_connect_returns_false(self) -> None:
        """UDP connect path is unimplemented; connect() must return False."""
        ctrl = _make_controller(CommunicationType.UDP)
        result = ctrl.connect(_make_config())
        assert result is False, "Unimplemented UDP connect must return False"

    def test_udp_is_not_connected_after_connect(self) -> None:
        """_is_connected must stay False after failed UDP connect."""
        ctrl = _make_controller(CommunicationType.UDP)
        ctrl.connect(_make_config())
        assert ctrl._is_connected is False

    def test_ethercat_connect_returns_false(self) -> None:
        """EtherCAT connect path is unimplemented; connect() must return False."""
        ctrl = _make_controller(CommunicationType.ETHERCAT)
        result = ctrl.connect(_make_config())
        assert result is False, "Unimplemented EtherCAT connect must return False"

    def test_ethercat_is_not_connected_after_connect(self) -> None:
        """_is_connected must stay False after failed EtherCAT connect."""
        ctrl = _make_controller(CommunicationType.ETHERCAT)
        ctrl.connect(_make_config())
        assert ctrl._is_connected is False

    def test_simulation_connect_still_succeeds(self) -> None:
        """SIMULATION connect must still return True (regression guard)."""
        ctrl = _make_controller(CommunicationType.SIMULATION)
        result = ctrl.connect(_make_config())
        assert result is True
        assert ctrl._is_connected is True

    def test_loopback_connect_still_succeeds(self) -> None:
        """LOOPBACK connect must still return True (regression guard)."""
        ctrl = _make_controller(CommunicationType.LOOPBACK)
        result = ctrl.connect(_make_config())
        assert result is True
        assert ctrl._is_connected is True


class TestHardwareReadStateStubs:
    """_read_state() must raise RuntimeError for hardware protocols instead of returning zeros."""

    def _controller_with_config(self, comm: CommunicationType) -> RealTimeController:
        ctrl = _make_controller(comm)
        ctrl._config = _make_config()
        ctrl._start_time = 0.0
        return ctrl

    def test_ros2_read_state_raises(self) -> None:
        """_read_state() for ROS2 must raise RuntimeError, not silently return zeros."""
        ctrl = self._controller_with_config(CommunicationType.ROS2)
        with pytest.raises(RuntimeError, match="not implemented"):
            ctrl._read_state()

    def test_udp_read_state_raises(self) -> None:
        """_read_state() for UDP must raise RuntimeError, not silently return zeros."""
        ctrl = self._controller_with_config(CommunicationType.UDP)
        with pytest.raises(RuntimeError, match="not implemented"):
            ctrl._read_state()

    def test_ethercat_read_state_raises(self) -> None:
        """_read_state() for EtherCAT must raise RuntimeError, not silently return zeros."""
        ctrl = self._controller_with_config(CommunicationType.ETHERCAT)
        with pytest.raises(RuntimeError, match="not implemented"):
            ctrl._read_state()

    def test_simulation_read_state_still_works(self) -> None:
        """_read_state() for SIMULATION must still return a valid RobotState (regression)."""
        ctrl = self._controller_with_config(CommunicationType.SIMULATION)
        state = ctrl._read_state()
        assert state.n_joints == 7
        np.testing.assert_array_equal(state.joint_positions, np.zeros(7))

    def test_loopback_read_state_still_works(self) -> None:
        """_read_state() for LOOPBACK must still return a valid RobotState (regression)."""
        ctrl = self._controller_with_config(CommunicationType.LOOPBACK)
        state = ctrl._read_state()
        assert state.n_joints == 7


class TestHardwareSendCommandStubs:
    """_send_command() must raise RuntimeError for hardware protocols instead of dropping commands."""

    def _make_torque_command(self, n_joints: int = 7) -> ControlCommand:
        return ControlCommand(
            timestamp=0.0,
            mode=ControlMode.TORQUE,
            torque_commands=np.ones(n_joints),
        )

    def _controller_with_config(self, comm: CommunicationType) -> RealTimeController:
        ctrl = _make_controller(comm)
        ctrl._config = _make_config()
        ctrl._start_time = 0.0
        return ctrl

    def test_ros2_send_command_raises(self) -> None:
        """_send_command() for ROS2 must raise RuntimeError, not drop commands silently."""
        ctrl = self._controller_with_config(CommunicationType.ROS2)
        with pytest.raises(RuntimeError, match="not implemented"):
            ctrl._send_command(self._make_torque_command())

    def test_udp_send_command_raises(self) -> None:
        """_send_command() for UDP must raise RuntimeError, not drop commands silently."""
        ctrl = self._controller_with_config(CommunicationType.UDP)
        with pytest.raises(RuntimeError, match="not implemented"):
            ctrl._send_command(self._make_torque_command())

    def test_ethercat_send_command_raises(self) -> None:
        """_send_command() for EtherCAT must raise RuntimeError, not drop commands silently."""
        ctrl = self._controller_with_config(CommunicationType.ETHERCAT)
        with pytest.raises(RuntimeError, match="not implemented"):
            ctrl._send_command(self._make_torque_command())

    def test_simulation_send_command_still_works(self) -> None:
        """_send_command() for SIMULATION must still succeed (regression guard)."""
        ctrl = self._controller_with_config(CommunicationType.SIMULATION)
        ctrl._send_command(self._make_torque_command())  # must not raise

    def test_loopback_send_command_still_works(self) -> None:
        """_send_command() for LOOPBACK must still update sim state (regression guard)."""
        ctrl = self._controller_with_config(CommunicationType.LOOPBACK)
        ctrl._send_command(self._make_torque_command())  # must not raise
