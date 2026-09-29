import numpy as np
import pytest

from src.deployment.teleoperation.devices import (
    BaseInputDevice,
    HapticDeviceInput,
    KeyboardMouseInput,
    SpaceMouseInput,
    VRControllerInput,
)
from src.shared.python.core.contracts.exceptions import StateError

pytestmark = pytest.mark.unit


class DummyDevice(BaseInputDevice):
    def update(self) -> None:
        self._require_connected("update")


def _assert_disconnected_device_contract(dev: BaseInputDevice) -> None:
    assert not dev.is_connected

    with pytest.raises(StateError, match=f"{type(dev).__name__} is not connected"):
        dev.get_pose()
    with pytest.raises(StateError, match=f"{type(dev).__name__} is not connected"):
        dev.get_twist()
    with pytest.raises(StateError, match=f"{type(dev).__name__} is not connected"):
        dev.get_gripper_state()
    with pytest.raises(StateError, match=f"{type(dev).__name__} is not connected"):
        dev.set_force_feedback(np.zeros(6))
    with pytest.raises(StateError, match=f"{type(dev).__name__} is not connected"):
        dev.get_buttons()
    with pytest.raises(StateError, match=f"{type(dev).__name__} is not connected"):
        dev.update()


def test_base_input_device() -> None:
    dev = DummyDevice()
    assert not dev.connect()
    _assert_disconnected_device_contract(dev)

    dev.disconnect()
    assert not dev.is_connected


def test_spacemouse_input_without_hardware_backend() -> None:
    dev = SpaceMouseInput(0)
    assert not dev.connect()

    dev.set_sensitivity(2.0)
    assert dev._sensitivity == 2.0
    _assert_disconnected_device_contract(dev)


def test_vr_controller_input_without_hardware_backend() -> None:
    dev = VRControllerInput("left", "steamvr")
    assert dev._hand == "left"
    assert dev._tracking_system == "steamvr"
    assert not dev.connect()

    assert dev.get_trigger_value() == 0.0
    assert dev.get_grip_value() == 0.0
    _assert_disconnected_device_contract(dev)


def test_haptic_device_input_without_hardware_backend() -> None:
    dev = HapticDeviceInput("phantom")
    assert not dev.connect()

    dev.set_workspace_scale(0.005)
    assert dev._workspace_scale == 0.005
    _assert_disconnected_device_contract(dev)


def test_keyboard_mouse_input() -> None:
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
    with pytest.raises(StateError, match="KeyboardMouseInput is not connected"):
        dev.update()
