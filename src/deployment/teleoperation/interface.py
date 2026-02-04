"""Teleoperation interface for robot control."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.deployment.realtime import ControlCommand
    from src.deployment.teleoperation.devices import InputDevice
    from src.engines.protocols import PhysicsEngineProtocol
    from src.learning.imitation import Demonstration


class TeleoperationMode(Enum):
    """Teleoperation control mode."""

    POSITION = "position"
    VELOCITY = "velocity"
    WRENCH = "wrench"
    IMPEDANCE = "impedance"


@dataclass
class WorkspaceMapping:
    """Workspace mapping configuration.

    Attributes:
        leader_frame: Leader (input device) reference frame.
        follower_frame: Follower (robot) reference frame.
        position_scale: Position scaling factor.
        rotation_scale: Rotation scaling factor.
        deadband: Deadband for small motions.
        rate_limit: Maximum velocity limit.
    """

    leader_frame: NDArray[np.floating] = field(
        default_factory=lambda: np.eye(4)
    )
    follower_frame: NDArray[np.floating] = field(
        default_factory=lambda: np.eye(4)
    )
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    deadband: float = 0.001
    rate_limit: float = 0.5


class TeleoperationInterface:
    """Interface for robot teleoperation.

    Provides mapping from input device commands to robot
    control commands, with support for:
    - Workspace mapping and scaling
    - Clutch for disengaging control
    - Haptic feedback
    - Demonstration recording

    Attributes:
        robot: Physics engine for robot model.
        input_device: Input device instance.
        mode: Current teleoperation mode.
    """

    def __init__(
        self,
        robot: "PhysicsEngineProtocol",
        input_device: "InputDevice",
    ) -> None:
        """Initialize teleoperation interface.

        Args:
            robot: Robot physics engine.
            input_device: Input device for commands.
        """
        self.robot = robot
        self.input = input_device
        self._mode = TeleoperationMode.POSITION
        self._workspace = WorkspaceMapping()
        self._clutch_engaged = True
        self._scaling = 1.0

        # Recording state
        self._recording = False
        self._recording_data: dict[str, list] = {}
        self._recording_start_time = 0.0

        # State tracking
        self._last_device_pose: NDArray[np.floating] | None = None
        self._reference_pose: NDArray[np.floating] | None = None

    @property
    def mode(self) -> TeleoperationMode:
        """Get current teleoperation mode."""
        return self._mode

    @property
    def is_clutch_engaged(self) -> bool:
        """Check if clutch is engaged (motion enabled)."""
        return self._clutch_engaged

    @property
    def is_recording(self) -> bool:
        """Check if demonstration recording is active."""
        return self._recording

    def set_control_mode(self, mode: TeleoperationMode) -> None:
        """Set teleoperation control mode.

        Args:
            mode: Teleoperation mode.
        """
        self._mode = mode

    def set_workspace_mapping(
        self,
        leader_frame: NDArray[np.floating],
        follower_frame: NDArray[np.floating],
        scaling: float = 1.0,
    ) -> None:
        """Configure workspace mapping between leader and follower.

        Args:
            leader_frame: Leader reference frame (4x4).
            follower_frame: Follower reference frame (4x4).
            scaling: Position scaling factor.
        """
        self._workspace.leader_frame = leader_frame
        self._workspace.follower_frame = follower_frame
        self._workspace.position_scale = scaling
        self._scaling = scaling

    def engage_clutch(self) -> None:
        """Engage clutch (enable motion)."""
        self._clutch_engaged = True
        # Set current pose as reference
        self._reference_pose = self.input.get_pose()
        self._last_device_pose = self._reference_pose.copy()

    def disengage_clutch(self) -> None:
        """Disengage clutch (disable motion)."""
        self._clutch_engaged = False

    def update(self) -> "ControlCommand":
        """Process input and generate control command.

        Returns:
            Control command for robot.
        """
        from src.deployment.realtime import ControlCommand, ControlMode

        # Get device input
        device_pose = self.input.get_pose()
        device_twist = self.input.get_twist()
        gripper = self.input.get_gripper_state()
        buttons = self.input.get_buttons()

        # Handle clutch from button
        if buttons.get("button_1", False):
            if not self._clutch_engaged:
                self.engage_clutch()
        else:
            if self._clutch_engaged and buttons.get("button_2", False):
                self.disengage_clutch()

        # Initialize reference if needed
        if self._reference_pose is None:
            self._reference_pose = device_pose.copy()
            self._last_device_pose = device_pose.copy()

        # Compute command based on mode
        if not self._clutch_engaged:
            # Return zero command when clutch disengaged
            n_joints = 7  # Default
            if hasattr(self.robot, "n_q"):
                n_joints = self.robot.n_q

            return ControlCommand(
                timestamp=time.perf_counter(),
                mode=ControlMode.TORQUE,
                torque_commands=np.zeros(n_joints),
                gripper_command=gripper,
            )

        if self._mode == TeleoperationMode.POSITION:
            return self._compute_position_command(device_pose, gripper)
        elif self._mode == TeleoperationMode.VELOCITY:
            return self._compute_velocity_command(device_twist, gripper)
        elif self._mode == TeleoperationMode.WRENCH:
            return self._compute_wrench_command(device_twist, gripper)
        elif self._mode == TeleoperationMode.IMPEDANCE:
            return self._compute_impedance_command(device_pose, gripper)

        # Default
        return ControlCommand(
            timestamp=time.perf_counter(),
            mode=ControlMode.TORQUE,
            torque_commands=np.zeros(7),
        )

    def _compute_position_command(
        self,
        device_pose: NDArray[np.floating],
        gripper: float,
    ) -> "ControlCommand":
        """Compute position control command.

        Args:
            device_pose: Device pose.
            gripper: Gripper command.

        Returns:
            Position control command.
        """
        from src.deployment.realtime import ControlCommand, ControlMode

        # Compute position delta from reference
        if self._reference_pose is not None:
            delta_pos = (
                device_pose[:3] - self._reference_pose[:3]
            ) * self._scaling

            # Apply deadband
            delta_norm = np.linalg.norm(delta_pos)
            if delta_norm < self._workspace.deadband:
                delta_pos = np.zeros(3)

            # Get current robot EE position
            if hasattr(self.robot, "get_ee_position"):
                current_pos = self.robot.get_ee_position()
            else:
                current_pos = np.zeros(3)

            target_pos = current_pos + delta_pos

            # Solve IK for joint positions
            if hasattr(self.robot, "solve_ik"):
                target_q, success = self.robot.solve_ik(
                    "end_effector",
                    np.concatenate([target_pos, device_pose[3:]]),
                )
            else:
                # Fallback: return zeros
                n_joints = 7
                if hasattr(self.robot, "n_q"):
                    n_joints = self.robot.n_q
                target_q = np.zeros(n_joints)

            # Update reference for relative control
            self._reference_pose = device_pose.copy()

            return ControlCommand(
                timestamp=time.perf_counter(),
                mode=ControlMode.POSITION,
                position_targets=target_q,
                gripper_command=gripper,
            )

        # Fallback
        return ControlCommand(
            timestamp=time.perf_counter(),
            mode=ControlMode.POSITION,
            position_targets=np.zeros(7),
            gripper_command=gripper,
        )

    def _compute_velocity_command(
        self,
        device_twist: NDArray[np.floating],
        gripper: float,
    ) -> "ControlCommand":
        """Compute velocity control command.

        Args:
            device_twist: Device twist.
            gripper: Gripper command.

        Returns:
            Velocity control command.
        """
        from src.deployment.realtime import ControlCommand, ControlMode

        # Scale twist
        scaled_twist = device_twist * self._scaling

        # Apply rate limit
        linear_norm = np.linalg.norm(scaled_twist[:3])
        if linear_norm > self._workspace.rate_limit:
            scaled_twist[:3] *= self._workspace.rate_limit / linear_norm

        # Map to joint velocities using Jacobian
        if hasattr(self.robot, "compute_jacobian"):
            J = self.robot.compute_jacobian("end_effector")
            # Pseudo-inverse for velocity mapping
            J_pinv = np.linalg.pinv(J)
            target_qd = J_pinv @ scaled_twist
        else:
            n_joints = 7
            if hasattr(self.robot, "n_q"):
                n_joints = self.robot.n_q
            target_qd = np.zeros(n_joints)

        return ControlCommand(
            timestamp=time.perf_counter(),
            mode=ControlMode.VELOCITY,
            velocity_targets=target_qd,
            gripper_command=gripper,
        )

    def _compute_wrench_command(
        self,
        device_twist: NDArray[np.floating],
        gripper: float,
    ) -> "ControlCommand":
        """Compute wrench/torque control command.

        Args:
            device_twist: Device twist (interpreted as force).
            gripper: Gripper command.

        Returns:
            Torque control command.
        """
        from src.deployment.realtime import ControlCommand, ControlMode

        # Interpret twist as desired wrench
        desired_wrench = device_twist * 10.0  # Scale to force

        # Map to joint torques using Jacobian transpose
        if hasattr(self.robot, "compute_jacobian"):
            J = self.robot.compute_jacobian("end_effector")
            target_tau = J.T @ desired_wrench
        else:
            n_joints = 7
            if hasattr(self.robot, "n_q"):
                n_joints = self.robot.n_q
            target_tau = np.zeros(n_joints)

        return ControlCommand(
            timestamp=time.perf_counter(),
            mode=ControlMode.TORQUE,
            torque_commands=target_tau,
            gripper_command=gripper,
        )

    def _compute_impedance_command(
        self,
        device_pose: NDArray[np.floating],
        gripper: float,
    ) -> "ControlCommand":
        """Compute impedance control command.

        Args:
            device_pose: Device pose.
            gripper: Gripper command.

        Returns:
            Impedance control command.
        """
        from src.deployment.realtime import ControlCommand, ControlMode

        # Get target position (similar to position mode)
        cmd = self._compute_position_command(device_pose, gripper)

        # Add impedance parameters
        n_joints = len(cmd.position_targets) if cmd.position_targets is not None else 7
        stiffness = np.ones(n_joints) * 100.0  # N/rad
        damping = np.ones(n_joints) * 10.0  # Ns/rad

        return ControlCommand(
            timestamp=time.perf_counter(),
            mode=ControlMode.IMPEDANCE,
            position_targets=cmd.position_targets,
            stiffness=stiffness,
            damping=damping,
            gripper_command=gripper,
        )

    def get_haptic_feedback(self) -> NDArray[np.floating]:
        """Get force feedback for haptic device.

        Computes appropriate force feedback based on
        robot contact forces and virtual fixtures.

        Returns:
            Wrench for haptic feedback.
        """
        feedback = np.zeros(6)

        # Get contact forces if available
        if hasattr(self.robot, "get_contact_forces"):
            contact_forces = self.robot.get_contact_forces()
            if contact_forces is not None:
                # Scale down for haptic device
                feedback[:3] = contact_forces[:3] * 0.1

        return feedback

    def start_demonstration_recording(self) -> None:
        """Begin recording demonstration data."""
        self._recording = True
        self._recording_start_time = time.perf_counter()
        self._recording_data = {
            "timestamps": [],
            "joint_positions": [],
            "joint_velocities": [],
            "actions": [],
            "gripper_commands": [],
        }

    def stop_demonstration_recording(self) -> "Demonstration":
        """Stop recording and return demonstration.

        Returns:
            Recorded demonstration.
        """
        from src.learning.imitation import Demonstration

        self._recording = False

        # Convert to arrays
        timestamps = np.array(self._recording_data["timestamps"])
        joint_positions = np.array(self._recording_data["joint_positions"])
        joint_velocities = np.array(self._recording_data["joint_velocities"])
        actions = np.array(self._recording_data["actions"])

        demo = Demonstration(
            timestamps=timestamps,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            actions=actions if len(actions) > 0 else None,
            source="teleoperation",
            success=True,
        )

        return demo

    def record_state(
        self,
        joint_positions: NDArray[np.floating],
        joint_velocities: NDArray[np.floating],
        action: NDArray[np.floating] | None = None,
    ) -> None:
        """Record current state for demonstration.

        Args:
            joint_positions: Current joint positions.
            joint_velocities: Current joint velocities.
            action: Applied action/torque.
        """
        if not self._recording:
            return

        timestamp = time.perf_counter() - self._recording_start_time
        self._recording_data["timestamps"].append(timestamp)
        self._recording_data["joint_positions"].append(joint_positions.copy())
        self._recording_data["joint_velocities"].append(joint_velocities.copy())
        if action is not None:
            self._recording_data["actions"].append(action.copy())
