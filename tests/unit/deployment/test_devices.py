import numpy as np

from src.deployment.teleoperation.devices import (
    BaseInputDevice,
    HapticDeviceInput,
    KeyboardMouseInput,
    SpaceMouseInput,
    VRControllerInput,
)


class DummyDevice(BaseInputDevice):
    def update(self):
        pass


def test_base_input_device():
    dev = DummyDevice()
    assert not dev.is_connected
    assert dev.connect()
    assert dev.is_connected

    pose = dev.get_pose()
    assert len(pose) == 7
    assert pose[3] == 1.0  # qw

    twist = dev.get_twist()
    assert len(twist) == 6
    assert np.all(twist == 0)

    assert dev.get_gripper_state() == 1.0

    dev.set_force_feedback(np.zeros(6))
    assert dev.get_buttons() == {}

    dev.disconnect()
    assert not dev.is_connected


def test_spacemouse_input():
    dev = SpaceMouseInput(0)
    assert not dev.is_connected
    assert dev.connect()
    assert dev.is_connected

    dev.update()
    dev.set_sensitivity(2.0)
    assert dev._sensitivity == 2.0

    buttons = dev.get_buttons()
    assert "button_1" in buttons

    # Test update when disconnected
    dev.disconnect()
    dev.update()


def test_vr_controller_input():
    dev = VRControllerInput("left", "steamvr")
    assert dev._hand == "left"
    assert dev._tracking_system == "steamvr"

    assert dev.connect()
    dev.update()

    assert dev.get_gripper_state() == 1.0
    assert dev.get_trigger_value() == 0.0
    assert dev.get_grip_value() == 0.0

    # Test update when disconnected
    dev.disconnect()
    dev.update()


def test_haptic_device_input():
    dev = HapticDeviceInput("phantom")
    assert dev.connect()

    dev.update()

    wrench = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    dev.set_force_feedback(wrench)

    dev.set_workspace_scale(0.005)
    assert dev._workspace_scale == 0.005

    dev.disconnect()
    dev.update()
    dev.set_force_feedback(wrench)


def test_keyboard_mouse_input():
    dev = KeyboardMouseInput()
    assert dev.connect()

    dev.set_key_state("forward", True)
    dev.set_key_state("up", True)
    dev.set_key_state("close_gripper", True)

    dev.update()

    twist = dev.get_twist()
    assert twist[0] > 0
    assert twist[2] > 0
    assert dev.get_gripper_state() == 0.0

    dev.set_key_state("open_gripper", True)
    dev.update()
    assert dev.get_gripper_state() == 1.0

    dev.disconnect()
    dev.update()
