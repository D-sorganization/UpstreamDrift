"""Comprehensive tests for deployment.teleoperation."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.deployment.realtime import ControlMode
from src.shared.python.core.contracts.exceptions import StateError
from src.deployment.teleoperation.devices import (
    HapticDeviceInput,
    KeyboardMouseInput,
    SpaceMouseInput,
    VRControllerInput,
)
from src.deployment.teleoperation.interface import (
    TeleoperationInterface,
    TeleoperationMode,
    WorkspaceMapping,
)

pytestmark = pytest.mark.unit


class FakeDevice:
    def __init__(self):
        self._pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self._twist = np.zeros(6)
        self._gripper = 0.5
        self._buttons: dict[str, bool] = {}

    def get_pose(self):
        return self._pose.copy()

    def get_twist(self):
        return self._twist.copy()

    def get_gripper_state(self):
        return self._gripper

    def set_force_feedback(self, w):
        pass

    def get_buttons(self):
        return dict(self._buttons)

    def update(self):
        pass


def _make_robot(n_q: int = 3, with_ik: bool = True, with_jac: bool = True):
    robot = MagicMock()
    robot.n_q = n_q
    if with_ik:
        robot.solve_ik.return_value = (np.ones(n_q), True)
    else:
        del robot.solve_ik
    if with_jac:
        robot.compute_jacobian.return_value = np.eye(6, n_q)
    else:
        del robot.compute_jacobian
    robot.get_ee_position.return_value = np.zeros(3)
    return robot


class TestWorkspaceMapping:
    def test_defaults(self) -> None:
        m = WorkspaceMapping()
        assert m.position_scale == 1.0


class TestTeleoperationInterface:
    def test_init(self) -> None:
        ti = TeleoperationInterface(_make_robot(), FakeDevice())
        assert ti.mode == TeleoperationMode.POSITION
        assert ti.is_clutch_engaged is True
        assert ti.is_recording is False

    def test_set_control_mode(self) -> None:
        ti = TeleoperationInterface(_make_robot(), FakeDevice())
        ti.set_control_mode(TeleoperationMode.VELOCITY)
        assert ti.mode == TeleoperationMode.VELOCITY

    def test_set_workspace_mapping(self) -> None:
        ti = TeleoperationInterface(_make_robot(), FakeDevice())
        ti.set_workspace_mapping(np.eye(4), np.eye(4), scaling=2.0)
        assert ti._scaling == 2.0

    def test_clutch_cycle(self) -> None:
        dev = FakeDevice()
        ti = TeleoperationInterface(_make_robot(), dev)
        ti.disengage_clutch()
        assert not ti.is_clutch_engaged
        ti.engage_clutch()
        assert ti.is_clutch_engaged
        assert ti._reference_pose is not None

    def test_update_clutch_disengaged_returns_zero(self) -> None:
        dev = FakeDevice()
        ti = TeleoperationInterface(_make_robot(), dev)
        ti.disengage_clutch()
        cmd = ti.update()
        assert cmd.mode == ControlMode.TORQUE
        assert np.all(cmd.torque_commands == 0)

    def test_update_button_engages_clutch(self) -> None:
        dev = FakeDevice()
        dev._buttons = {"button_1": True}
        ti = TeleoperationInterface(_make_robot(), dev)
        ti.disengage_clutch()
        ti.update()
        assert ti.is_clutch_engaged

    def test_update_button2_disengages(self) -> None:
        dev = FakeDevice()
        dev._buttons = {"button_2": True}
        ti = TeleoperationInterface(_make_robot(), dev)
        ti.update()
        assert not ti.is_clutch_engaged

    def test_update_position_mode(self) -> None:
        ti = TeleoperationInterface(_make_robot(), FakeDevice())
        cmd = ti.update()
        assert cmd.mode == ControlMode.POSITION

    def test_update_velocity_mode(self) -> None:
        dev = FakeDevice()
        dev._twist = np.array([0.1, 0, 0, 0, 0, 0])
        ti = TeleoperationInterface(_make_robot(), dev)
        ti.set_control_mode(TeleoperationMode.VELOCITY)
        cmd = ti.update()
        assert cmd.mode == ControlMode.VELOCITY

    def test_update_velocity_rate_limited(self) -> None:
        dev = FakeDevice()
        dev._twist = np.array([10.0, 0, 0, 0, 0, 0])
        ti = TeleoperationInterface(_make_robot(), dev)
        ti.set_control_mode(TeleoperationMode.VELOCITY)
        cmd = ti.update()
        assert cmd.velocity_targets is not None

    def test_update_velocity_no_jacobian(self) -> None:
        ti = TeleoperationInterface(_make_robot(with_jac=False), FakeDevice())
        ti.set_control_mode(TeleoperationMode.VELOCITY)
        cmd = ti.update()
        assert cmd.velocity_targets is not None

    def test_update_wrench_mode(self) -> None:
        ti = TeleoperationInterface(_make_robot(), FakeDevice())
        ti.set_control_mode(TeleoperationMode.WRENCH)
        cmd = ti.update()
        assert cmd.mode == ControlMode.TORQUE

    def test_update_wrench_no_jacobian(self) -> None:
        ti = TeleoperationInterface(_make_robot(with_jac=False), FakeDevice())
        ti.set_control_mode(TeleoperationMode.WRENCH)
        cmd = ti.update()
        assert cmd.torque_commands is not None

    def test_update_impedance_mode(self) -> None:
        ti = TeleoperationInterface(_make_robot(), FakeDevice())
        ti.set_control_mode(TeleoperationMode.IMPEDANCE)
        cmd = ti.update()
        assert cmd.mode == ControlMode.IMPEDANCE
        assert cmd.stiffness is not None

    def test_update_position_no_ik(self) -> None:
        ti = TeleoperationInterface(_make_robot(with_ik=False), FakeDevice())
        cmd = ti.update()
        assert cmd.position_targets is not None

    def test_haptic_feedback_default(self) -> None:
        robot = MagicMock()
        del robot.get_contact_forces
        ti = TeleoperationInterface(robot, FakeDevice())
        assert np.all(ti.get_haptic_feedback() == 0)

    def test_haptic_feedback_with_contact(self) -> None:
        robot = _make_robot()
        robot.get_contact_forces.return_value = np.array([10.0, 0, 0, 0, 0, 0])
        ti = TeleoperationInterface(robot, FakeDevice())
        fb = ti.get_haptic_feedback()
        assert fb[0] == pytest.approx(1.0)

    def test_haptic_feedback_none(self) -> None:
        robot = _make_robot()
        robot.get_contact_forces.return_value = None
        ti = TeleoperationInterface(robot, FakeDevice())
        assert np.all(ti.get_haptic_feedback() == 0)

    def test_recording_cycle(self) -> None:
        ti = TeleoperationInterface(_make_robot(), FakeDevice())
        ti.start_demonstration_recording()
        assert ti.is_recording
        ti.record_state(np.zeros(3), np.zeros(3), np.zeros(3))
        ti.record_state(np.ones(3), np.ones(3))
        demo = ti.stop_demonstration_recording()
        assert not ti.is_recording
        assert demo.source == "teleoperation"
        assert len(demo.timestamps) == 2

    def test_record_state_when_not_recording(self) -> None:
        ti = TeleoperationInterface(_make_robot(), FakeDevice())
        ti.record_state(np.zeros(3), np.zeros(3))
        # No exception, nothing recorded


class TestInputDevices:
    """Contract for the input-device stubs after #7360.

    These tests previously asserted the pre-#7360 behaviour, where `connect()`
    faked success and `update()` silently no-opped, so they were red on `main`
    (#8036). The three hardware devices have no driver behind them (#8058), and
    the honest contract is: `connect()` returns False and every state query
    raises `StateError`. Implementing any device will fail these tests, which is
    the intent -- they must be rewritten with the implementation.
    """

    def test_spacemouse_reports_no_hardware_and_refuses_state_queries(self) -> None:
        d = SpaceMouseInput(0)

        with pytest.raises(
            NotImplementedError, match="Hardware driver not implemented"
        ):
            d.connect()

        for operation in (d.update, d.get_twist, d.get_pose, d.get_buttons):
            with pytest.raises(StateError):
                operation()

        # Pure configuration that touches no hardware is still allowed.
        d.set_sensitivity(0.5)

    def test_vr_controller_reports_no_hardware_and_refuses_state_queries(self) -> None:
        d = VRControllerInput("right", "steamvr")

        with pytest.raises(
            NotImplementedError, match="Hardware driver not implemented"
        ):
            d.connect()

        with pytest.raises(StateError):
            d.update()

        # Cached scalar accessors do not require a connection.
        assert d.get_trigger_value() == 0.0
        assert d.get_grip_value() == 0.0

    def test_haptic_reports_no_hardware_and_refuses_state_queries(self) -> None:
        d = HapticDeviceInput("phantom")

        with pytest.raises(
            NotImplementedError, match="Hardware driver not implemented"
        ):
            d.connect()

        with pytest.raises(StateError):
            d.update()

        # Commanding force into hardware that is not there must be refused, not
        # silently dropped -- this is a physical-safety surface.
        with pytest.raises(StateError):
            d.set_force_feedback(np.array([1, 2, 3, 0, 0, 0], dtype=float))

        d.disconnect()
        with pytest.raises(StateError):
            d.set_force_feedback(np.zeros(6))

        # Pure configuration is still allowed.
        d.set_workspace_scale(0.002)

    def test_keyboard_connects_and_drives_twist(self) -> None:
        d = KeyboardMouseInput()

        # KeyboardMouseInput needs no hardware, so it is the one device that
        # genuinely connects.
        assert d.connect() is True

        d.set_key_state("forward", True)
        d.set_key_state("close_gripper", True)
        d.update()
        assert d.get_twist()[0] > 0

        d.set_key_state("open_gripper", True)
        d.update()
        assert d.get_gripper_state() == 1.0

        d.set_key_state("unknown_key", True)  # noop

    def test_keyboard_refuses_update_after_disconnect(self) -> None:
        d = KeyboardMouseInput()
        d.connect()
        d.disconnect()

        with pytest.raises(StateError):
            d.update()
