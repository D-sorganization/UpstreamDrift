"""Tests for teleoperation module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.deployment.realtime import ControlMode
from src.deployment.teleoperation.devices import (
    HapticDeviceInput,
    KeyboardMouseInput,
    SpaceMouseInput,
    VRControllerInput,
)
from src.deployment.teleoperation.interface import (
    TeleoperationInterface,
    TeleoperationMode,
)


@pytest.fixture
def mock_robot() -> MagicMock:
    """Mock robot physics engine."""
    robot = MagicMock()
    robot.n_q = 7
    robot.get_ee_position.return_value = np.zeros(3)
    robot.solve_ik.return_value = (np.zeros(7), True)
    robot.compute_jacobian.return_value = np.eye(6, 7)
    return robot


@pytest.fixture
def mock_device() -> MagicMock:
    """Mock input device."""
    device = MagicMock()
    device.get_pose.return_value = np.array([1, 2, 3, 1, 0, 0, 0], dtype=np.float64)
    device.get_twist.return_value = np.ones(6, dtype=np.float64)
    device.get_gripper_state.return_value = 1.0
    device.get_buttons.return_value = {"button_1": False, "button_2": False}
    return device


def test_spacemouse_input() -> None:
    """Test SpaceMouseInput."""
    mouse = SpaceMouseInput(device_index=0)
    assert not mouse.is_connected
    assert mouse.connect()
    assert mouse.is_connected
    mouse.update()
    mouse.set_sensitivity(2.0)
    assert mouse._sensitivity == 2.0


def test_vr_controller_input() -> None:
    """Test VRControllerInput."""
    vr = VRControllerInput(hand="right")
    assert vr.connect()
    vr.update()
    assert vr.get_gripper_state() == 1.0
    assert vr.get_trigger_value() == 0.0
    assert vr.get_grip_value() == 0.0


def test_haptic_device_input() -> None:
    """Test HapticDeviceInput."""
    haptic = HapticDeviceInput()
    assert haptic.connect()
    haptic.update()
    haptic.set_workspace_scale(0.01)

    # Test setting feedback
    haptic.set_force_feedback(np.ones(6))

    haptic.disconnect()
    haptic.set_force_feedback(np.zeros(6))  # No op when disconnected


def test_keyboard_input() -> None:
    """Test KeyboardMouseInput."""
    kb = KeyboardMouseInput()
    assert kb.connect()

    # Set keys
    kb.set_key_state("forward", True)
    kb.set_key_state("open_gripper", True)

    kb.update()
    twist = kb.get_twist()
    assert twist[0] > 0  # vx > 0
    assert kb.get_gripper_state() == 1.0

    kb.set_key_state("close_gripper", True)
    kb.set_key_state("open_gripper", False)
    kb.update()
    assert kb.get_gripper_state() == 0.0


def test_teleop_interface_clutch(mock_robot: MagicMock, mock_device: MagicMock) -> None:
    """Test teleoperation clutch."""
    interface = TeleoperationInterface(mock_robot, mock_device)
    assert interface.is_clutch_engaged

    interface.disengage_clutch()
    assert not interface.is_clutch_engaged

    # Try updating with disengaged clutch
    cmd = interface.update()
    assert cmd.mode == ControlMode.TORQUE
    assert np.allclose(cmd.torque_commands, np.zeros(7))

    # Try engaging via button
    mock_device.get_buttons.return_value = {"button_1": True}
    interface.update()
    assert interface.is_clutch_engaged


def test_teleop_modes(mock_robot: MagicMock, mock_device: MagicMock) -> None:
    """Test different teleoperation modes."""
    interface = TeleoperationInterface(mock_robot, mock_device)

    # Initial trigger to set reference pose
    interface.update()

    # Change pose so delta is non-zero
    mock_device.get_pose.return_value = np.array(
        [2, 5, 3, 1, 0, 0, 0], dtype=np.float64
    )

    # POSITION mode
    interface.set_control_mode(TeleoperationMode.POSITION)
    cmd = interface.update()
    assert cmd.mode == ControlMode.POSITION
    assert cmd.position_targets is not None

    # VELOCITY mode
    interface.set_control_mode(TeleoperationMode.VELOCITY)
    cmd = interface.update()
    assert cmd.mode == ControlMode.VELOCITY
    assert cmd.velocity_targets is not None

    # WRENCH mode
    interface.set_control_mode(TeleoperationMode.WRENCH)
    cmd = interface.update()
    assert cmd.mode == ControlMode.TORQUE
    assert cmd.torque_commands is not None

    # IMPEDANCE mode
    interface.set_control_mode(TeleoperationMode.IMPEDANCE)
    cmd = interface.update()
    assert cmd.mode == ControlMode.IMPEDANCE
    assert cmd.position_targets is not None
    assert cmd.stiffness is not None


def test_teleop_demonstration_recording(
    mock_robot: MagicMock, mock_device: MagicMock
) -> None:
    """Test teleoperation recording."""
    interface = TeleoperationInterface(mock_robot, mock_device)

    assert not interface.is_recording
    interface.start_demonstration_recording()
    assert interface.is_recording

    # Record some states
    interface.record_state(np.zeros(7), np.zeros(7), np.ones(7))
    interface.record_state(np.ones(7), np.ones(7), np.ones(7))

    demo = interface.stop_demonstration_recording()
    assert not interface.is_recording

    assert len(demo.joint_positions) == 2
    assert demo.success is True
    assert demo.source == "teleoperation"


def test_workspace_mapping_and_feedback(
    mock_robot: MagicMock, mock_device: MagicMock
) -> None:
    """Test mapping utilities."""
    interface = TeleoperationInterface(mock_robot, mock_device)

    interface.set_workspace_mapping(
        leader_frame=np.eye(4), follower_frame=np.eye(4), scaling=2.0
    )

    assert interface._scaling == 2.0

    # Force feedback
    mock_robot.get_contact_forces.return_value = np.array([10, 0, 0, 0, 0, 0])
    fb = interface.get_haptic_feedback()
    assert fb[0] == 1.0  # scaled by 0.1
