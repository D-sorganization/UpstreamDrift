import numpy as np

from src.deployment.realtime import ControlMode
from src.deployment.teleoperation.interface import (
    TeleoperationInterface,
    TeleoperationMode,
)


class MockRobot:
    def __init__(self):
        self.n_q = 7
        self.ee_pos = np.zeros(3)
        self.contact_forces = np.zeros(6)

    def get_ee_position(self):
        return self.ee_pos

    def solve_ik(self, link, target):
        return np.ones(7), True

    def compute_jacobian(self, link):
        return np.eye(6, 7)

    def get_contact_forces(self):
        return self.contact_forces


class MockInputDevice:
    def __init__(self):
        self.pose = np.zeros(7)
        self.pose[3] = 1.0
        self.twist = np.zeros(6)
        self.gripper = 1.0
        self.buttons = {}

    def get_pose(self):
        return self.pose

    def get_twist(self):
        return self.twist

    def get_gripper_state(self):
        return self.gripper

    def get_buttons(self):
        return self.buttons


def test_teleoperation_interface_init():
    robot = MockRobot()
    device = MockInputDevice()
    interface = TeleoperationInterface(robot, device)

    assert interface.mode == TeleoperationMode.POSITION
    assert interface.is_clutch_engaged
    assert not interface.is_recording


def test_teleoperation_set_workspace_mapping():
    robot = MockRobot()
    device = MockInputDevice()
    interface = TeleoperationInterface(robot, device)

    interface.set_workspace_mapping(np.eye(4) * 2, np.eye(4) * 3, scaling=2.0)
    assert interface._scaling == 2.0
    assert interface._workspace.position_scale == 2.0


def test_clutch_control():
    robot = MockRobot()
    device = MockInputDevice()
    interface = TeleoperationInterface(robot, device)

    interface.disengage_clutch()
    assert not interface.is_clutch_engaged

    cmd = interface.update()
    assert cmd.mode == ControlMode.TORQUE
    assert np.all(cmd.torque_commands == 0)

    interface.engage_clutch()
    assert interface.is_clutch_engaged


def test_update_position_mode():
    robot = MockRobot()
    device = MockInputDevice()
    interface = TeleoperationInterface(robot, device)
    interface.set_control_mode(TeleoperationMode.POSITION)

    # Initial reference set
    interface.update()

    # Move device
    device.pose[0] = 0.5
    cmd = interface.update()

    assert cmd.mode == ControlMode.POSITION
    assert cmd.position_targets is not None
    assert cmd.gripper_command == 1.0


def test_update_velocity_mode():
    robot = MockRobot()
    device = MockInputDevice()
    interface = TeleoperationInterface(robot, device)
    interface.set_control_mode(TeleoperationMode.VELOCITY)

    device.twist[0] = 0.1
    cmd = interface.update()

    assert cmd.mode == ControlMode.VELOCITY
    assert cmd.velocity_targets is not None


def test_update_wrench_mode():
    robot = MockRobot()
    device = MockInputDevice()
    interface = TeleoperationInterface(robot, device)
    interface.set_control_mode(TeleoperationMode.WRENCH)

    device.twist[0] = 0.1
    cmd = interface.update()

    assert cmd.mode == ControlMode.TORQUE
    assert cmd.torque_commands is not None


def test_update_impedance_mode():
    robot = MockRobot()
    device = MockInputDevice()
    interface = TeleoperationInterface(robot, device)
    interface.set_control_mode(TeleoperationMode.IMPEDANCE)

    cmd = interface.update()

    assert cmd.mode == ControlMode.IMPEDANCE
    assert cmd.position_targets is not None
    assert cmd.stiffness is not None


def test_get_haptic_feedback():
    robot = MockRobot()
    device = MockInputDevice()
    interface = TeleoperationInterface(robot, device)

    robot.contact_forces = np.array([10, 20, 30, 0, 0, 0], dtype=np.float64)
    feedback = interface.get_haptic_feedback()
    assert feedback[0] == 1.0
    assert feedback[1] == 2.0


def test_demonstration_recording():
    robot = MockRobot()
    device = MockInputDevice()
    interface = TeleoperationInterface(robot, device)

    interface.start_demonstration_recording()
    assert interface.is_recording

    interface.record_state(np.ones(7), np.ones(7), np.ones(7))
    interface.record_state(np.zeros(7), np.zeros(7), np.zeros(7))

    demo = interface.stop_demonstration_recording()
    assert not interface.is_recording
    assert len(demo.timestamps) == 2
    assert len(demo.actions) == 2
    assert demo.source == "teleoperation"
    assert demo.success
