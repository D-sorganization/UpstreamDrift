"""Comprehensive tests for deployment.realtime.state."""

from __future__ import annotations

import numpy as np
import pytest

from src.deployment.realtime.state import (
    ControlCommand,
    ControlMode,
    IMUReading,
    RobotState,
)


class TestIMUReading:
    def test_valid_imu(self) -> None:
        imu = IMUReading(
            timestamp=1.0,
            linear_acceleration=np.zeros(3),
            angular_velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        assert imu.timestamp == 1.0

    def test_imu_no_orientation(self) -> None:
        imu = IMUReading(
            timestamp=0.0,
            linear_acceleration=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        assert imu.orientation is None

    def test_imu_bad_linear_acc(self) -> None:
        with pytest.raises(ValueError, match="linear_acceleration"):
            IMUReading(
                timestamp=0.0,
                linear_acceleration=np.zeros(4),
                angular_velocity=np.zeros(3),
            )

    def test_imu_bad_angular_vel(self) -> None:
        with pytest.raises(ValueError, match="angular_velocity"):
            IMUReading(
                timestamp=0.0,
                linear_acceleration=np.zeros(3),
                angular_velocity=np.zeros(2),
            )

    def test_imu_bad_orientation(self) -> None:
        with pytest.raises(ValueError, match="orientation"):
            IMUReading(
                timestamp=0.0,
                linear_acceleration=np.zeros(3),
                angular_velocity=np.zeros(3),
                orientation=np.zeros(3),
            )


class TestRobotState:
    def _make(self) -> RobotState:
        return RobotState(
            timestamp=0.5,
            joint_positions=np.arange(7.0),
            joint_velocities=np.ones(7),
            joint_torques=np.zeros(7),
            ft_wrenches={"wrist": np.arange(6.0)},
        )

    def test_get_ft_wrench_present(self) -> None:
        s = self._make()
        w = s.get_ft_wrench("wrist")
        assert w is not None
        np.testing.assert_array_equal(w, np.arange(6.0))

    def test_get_ft_wrench_missing(self) -> None:
        s = self._make()
        assert s.get_ft_wrench("ankle") is None

    def test_get_ft_wrench_no_dict(self) -> None:
        s = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            joint_torques=np.zeros(7),
        )
        assert s.get_ft_wrench("wrist") is None

    def test_get_ft_wrench_none_name(self) -> None:
        s = self._make()
        with pytest.raises(ValueError, match="sensor_name"):
            s.get_ft_wrench(None)  # type: ignore[arg-type]


class TestControlCommandFactories:
    def test_position_command(self) -> None:
        cmd = ControlCommand.position_command(0.0, np.zeros(7), np.ones(7))
        assert cmd.mode == ControlMode.POSITION
        assert cmd.position_targets is not None
        assert cmd.feedforward_torque is not None

    def test_torque_command(self) -> None:
        cmd = ControlCommand.torque_command(0.0, np.zeros(7))
        assert cmd.mode == ControlMode.TORQUE

    def test_impedance_command(self) -> None:
        cmd = ControlCommand.impedance_command(
            0.0, np.zeros(7), np.ones(7) * 100, np.ones(7) * 10
        )
        assert cmd.mode == ControlMode.IMPEDANCE
        assert cmd.stiffness is not None
        assert cmd.damping is not None


class TestControlCommandValidate:
    def test_validate_position_ok(self) -> None:
        cmd = ControlCommand.position_command(0.0, np.zeros(7))
        assert cmd.validate(7) is True

    def test_validate_position_missing(self) -> None:
        cmd = ControlCommand(timestamp=0.0, mode=ControlMode.POSITION)
        with pytest.raises(ValueError, match="position_targets"):
            cmd.validate(7)

    def test_validate_position_wrong_len(self) -> None:
        cmd = ControlCommand.position_command(0.0, np.zeros(5))
        with pytest.raises(ValueError, match="position_targets length"):
            cmd.validate(7)

    def test_validate_velocity_ok(self) -> None:
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.VELOCITY,
            velocity_targets=np.zeros(7),
        )
        assert cmd.validate(7) is True

    def test_validate_velocity_missing(self) -> None:
        cmd = ControlCommand(timestamp=0.0, mode=ControlMode.VELOCITY)
        with pytest.raises(ValueError, match="velocity_targets"):
            cmd.validate(7)

    def test_validate_velocity_wrong_len(self) -> None:
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.VELOCITY,
            velocity_targets=np.zeros(3),
        )
        with pytest.raises(ValueError, match="velocity_targets length"):
            cmd.validate(7)

    def test_validate_torque_ok(self) -> None:
        cmd = ControlCommand.torque_command(0.0, np.zeros(7))
        assert cmd.validate(7) is True

    def test_validate_torque_missing(self) -> None:
        cmd = ControlCommand(timestamp=0.0, mode=ControlMode.TORQUE)
        with pytest.raises(ValueError, match="torque_commands"):
            cmd.validate(7)

    def test_validate_torque_wrong_len(self) -> None:
        cmd = ControlCommand.torque_command(0.0, np.zeros(2))
        with pytest.raises(ValueError, match="torque_commands length"):
            cmd.validate(7)

    def test_validate_impedance_ok(self) -> None:
        cmd = ControlCommand.impedance_command(0.0, np.zeros(7), np.ones(7), np.ones(7))
        assert cmd.validate(7) is True

    def test_validate_impedance_missing_positions(self) -> None:
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.IMPEDANCE,
            stiffness=np.ones(7),
            damping=np.ones(7),
        )
        with pytest.raises(ValueError, match="position_targets"):
            cmd.validate(7)

    def test_validate_impedance_missing_stiffness(self) -> None:
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.IMPEDANCE,
            position_targets=np.zeros(7),
        )
        with pytest.raises(ValueError, match="stiffness and damping"):
            cmd.validate(7)

    def test_validate_hybrid_passes(self) -> None:
        cmd = ControlCommand(timestamp=0.0, mode=ControlMode.HYBRID)
        assert cmd.validate(7) is True
