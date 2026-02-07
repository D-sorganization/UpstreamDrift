"""State definitions for real-time robot control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ControlMode(Enum):
    """Robot control mode."""

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    IMPEDANCE = "impedance"
    HYBRID = "hybrid"


@dataclass
class IMUReading:
    """IMU sensor reading.

    Attributes:
        timestamp: Measurement timestamp.
        linear_acceleration: Linear acceleration [ax, ay, az] (m/s²).
        angular_velocity: Angular velocity [wx, wy, wz] (rad/s).
        orientation: Orientation quaternion [w, x, y, z].
    """

    timestamp: float
    linear_acceleration: NDArray[np.floating]
    angular_velocity: NDArray[np.floating]
    orientation: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        """Validate IMU reading dimensions."""
        if self.linear_acceleration.shape != (3,):
            raise ValueError("linear_acceleration must have shape (3,)")
        if self.angular_velocity.shape != (3,):
            raise ValueError("angular_velocity must have shape (3,)")
        if self.orientation is not None and self.orientation.shape != (4,):
            raise ValueError("orientation must have shape (4,)")


@dataclass
class RobotState:
    """State received from real robot.

    Contains all sensor data from the robot at a single timestep.

    Attributes:
        timestamp: State measurement timestamp (seconds).
        joint_positions: Joint positions in radians.
        joint_velocities: Joint velocities in rad/s.
        joint_torques: Measured joint torques in Nm.
        ft_wrenches: Force/torque sensor readings by name.
        imu_data: IMU sensor reading.
        contact_states: Binary contact states for each foot/end-effector.
        motor_temperatures: Motor temperatures in Celsius.
        battery_level: Battery level (0.0 to 1.0).
    """

    timestamp: float
    joint_positions: NDArray[np.floating]
    joint_velocities: NDArray[np.floating]
    joint_torques: NDArray[np.floating]
    ft_wrenches: dict[str, NDArray[np.floating]] | None = None
    imu_data: IMUReading | None = None
    contact_states: list[bool] | None = None
    motor_temperatures: NDArray[np.floating] | None = None
    battery_level: float | None = None

    @property
    def n_joints(self) -> int:
        """Number of joints."""
        return len(self.joint_positions)

    def get_state_vector(self) -> NDArray[np.floating]:
        """Get concatenated state vector [q, qd].

        Returns:
            State vector of shape (2*n_joints,).
        """
        return np.concatenate([self.joint_positions, self.joint_velocities])

    def get_ft_wrench(self, sensor_name: str) -> NDArray[np.floating] | None:
        """Get force/torque wrench for a sensor.

        Args:
            sensor_name: Name of the F/T sensor.

        Returns:
            Wrench [fx, fy, fz, tx, ty, tz] or None if not available.
        """
        if self.ft_wrenches is None:
            return None
        return self.ft_wrenches.get(sensor_name)


@dataclass
class ControlCommand:
    """Command sent to real robot.

    The command type depends on the control mode:
    - POSITION: position_targets required
    - VELOCITY: velocity_targets required
    - TORQUE: torque_commands required
    - IMPEDANCE: position_targets + stiffness + damping

    Attributes:
        timestamp: Command timestamp.
        mode: Control mode for this command.
        position_targets: Target joint positions (rad).
        velocity_targets: Target joint velocities (rad/s).
        torque_commands: Commanded joint torques (Nm).
        feedforward_torque: Feed-forward torque term (Nm).
        stiffness: Joint stiffness for impedance control.
        damping: Joint damping for impedance control.
        gripper_command: Gripper position command (0=closed, 1=open).
    """

    timestamp: float
    mode: ControlMode
    position_targets: NDArray[np.floating] | None = None
    velocity_targets: NDArray[np.floating] | None = None
    torque_commands: NDArray[np.floating] | None = None
    feedforward_torque: NDArray[np.floating] | None = None
    stiffness: NDArray[np.floating] | None = None
    damping: NDArray[np.floating] | None = None
    gripper_command: float | None = None

    def validate(self, n_joints: int) -> bool:
        """Validate command dimensions and mode consistency.

        Args:
            n_joints: Expected number of joints.

        Returns:
            True if command is valid.

        Raises:
            ValueError: If command is invalid.
        """
        if self.mode == ControlMode.POSITION:
            if self.position_targets is None:
                raise ValueError("Position mode requires position_targets")
            if len(self.position_targets) != n_joints:
                raise ValueError(
                    f"position_targets length ({len(self.position_targets)}) "
                    f"must match n_joints ({n_joints})"
                )

        elif self.mode == ControlMode.VELOCITY:
            if self.velocity_targets is None:
                raise ValueError("Velocity mode requires velocity_targets")
            if len(self.velocity_targets) != n_joints:
                raise ValueError(
                    f"velocity_targets length ({len(self.velocity_targets)}) "
                    f"must match n_joints ({n_joints})"
                )

        elif self.mode == ControlMode.TORQUE:
            if self.torque_commands is None:
                raise ValueError("Torque mode requires torque_commands")
            if len(self.torque_commands) != n_joints:
                raise ValueError(
                    f"torque_commands length ({len(self.torque_commands)}) "
                    f"must match n_joints ({n_joints})"
                )

        elif self.mode == ControlMode.IMPEDANCE:
            if self.position_targets is None:
                raise ValueError("Impedance mode requires position_targets")
            if self.stiffness is None or self.damping is None:
                raise ValueError("Impedance mode requires stiffness and damping")

        return True

    @classmethod
    def position_command(
        cls,
        timestamp: float,
        positions: NDArray[np.floating],
        feedforward: NDArray[np.floating] | None = None,
    ) -> ControlCommand:
        """Create a position control command.

        Args:
            timestamp: Command timestamp.
            positions: Target joint positions.
            feedforward: Optional feed-forward torques.

        Returns:
            Position control command.
        """
        return cls(
            timestamp=timestamp,
            mode=ControlMode.POSITION,
            position_targets=positions,
            feedforward_torque=feedforward,
        )

    @classmethod
    def torque_command(
        cls,
        timestamp: float,
        torques: NDArray[np.floating],
    ) -> ControlCommand:
        """Create a torque control command.

        Args:
            timestamp: Command timestamp.
            torques: Commanded joint torques.

        Returns:
            Torque control command.
        """
        return cls(
            timestamp=timestamp,
            mode=ControlMode.TORQUE,
            torque_commands=torques,
        )

    @classmethod
    def impedance_command(
        cls,
        timestamp: float,
        positions: NDArray[np.floating],
        stiffness: NDArray[np.floating],
        damping: NDArray[np.floating],
        feedforward: NDArray[np.floating] | None = None,
    ) -> ControlCommand:
        """Create an impedance control command.

        Args:
            timestamp: Command timestamp.
            positions: Target joint positions.
            stiffness: Joint stiffness values.
            damping: Joint damping values.
            feedforward: Optional feed-forward torques.

        Returns:
            Impedance control command.
        """
        return cls(
            timestamp=timestamp,
            mode=ControlMode.IMPEDANCE,
            position_targets=positions,
            stiffness=stiffness,
            damping=damping,
            feedforward_torque=feedforward,
        )
