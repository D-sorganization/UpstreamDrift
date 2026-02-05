"""Safety monitoring for robot operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.deployment.realtime import ControlCommand, RobotConfig, RobotState


class SafetyStatusLevel(Enum):
    """Safety status levels."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    EMERGENCY = "emergency"


@dataclass
class SafetyStatus:
    """Current safety status.

    Attributes:
        level: Overall safety level.
        is_safe: Whether operation is safe.
        violations: List of safety violations.
        warnings: List of safety warnings.
        speed_override: Applied speed reduction (0-1).
    """

    level: SafetyStatusLevel
    is_safe: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    speed_override: float = 1.0


@dataclass
class SafetyLimits:
    """Safety limits for robot operation.

    Follows ISO 10218-1 and ISO/TS 15066 guidelines for
    collaborative robots.

    Attributes:
        max_joint_velocity: Maximum joint velocities (rad/s).
        max_joint_torque: Maximum joint torques (Nm).
        max_cartesian_velocity: Maximum end-effector velocity (m/s).
        max_cartesian_force: Maximum contact force (N).
        workspace_bounds: Workspace limits [x_min, x_max, y_min, ...].
        forbidden_zones: List of forbidden zone boxes.
        max_contact_force: Maximum human contact force (N, ISO 10218-1).
        max_pressure: Maximum contact pressure (N/cm², ISO/TS 15066).
        joint_limits_lower: Lower joint position limits (rad).
        joint_limits_upper: Upper joint position limits (rad).
    """

    max_joint_velocity: NDArray[np.floating]
    max_joint_torque: NDArray[np.floating]
    max_cartesian_velocity: float = 0.25  # m/s for collaborative mode
    max_cartesian_force: float = 150.0
    workspace_bounds: NDArray[np.floating] | None = None
    forbidden_zones: list[NDArray[np.floating]] = field(default_factory=list)
    max_contact_force: float = 150.0  # ISO 10218-1
    max_pressure: float = 110.0  # N/cm² ISO/TS 15066
    joint_limits_lower: NDArray[np.floating] | None = None
    joint_limits_upper: NDArray[np.floating] | None = None

    @classmethod
    def from_config(cls, robot_config: RobotConfig) -> SafetyLimits:
        """Create safety limits from robot configuration.

        Args:
            robot_config: Robot configuration.

        Returns:
            Safety limits instance.
        """
        n_joints = robot_config.n_joints

        # Default limits
        max_vel = (
            robot_config.velocity_limits
            if robot_config.velocity_limits is not None
            else np.ones(n_joints) * 2.0  # rad/s
        )
        max_torque = (
            robot_config.torque_limits
            if robot_config.torque_limits is not None
            else np.ones(n_joints) * 50.0  # Nm
        )

        return cls(
            max_joint_velocity=max_vel,
            max_joint_torque=max_torque,
            joint_limits_lower=robot_config.joint_limits_lower,
            joint_limits_upper=robot_config.joint_limits_upper,
        )


class SafetyMonitor:
    """Real-time safety monitoring for robot operation.

    Monitors robot state and commands for safety violations,
    enforcing limits on velocities, torques, and workspace.

    Attributes:
        config: Robot configuration.
        limits: Safety limits.
        status: Current safety status.
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        limits: SafetyLimits | None = None,
    ) -> None:
        """Initialize safety monitor.

        Args:
            robot_config: Robot configuration.
            limits: Safety limits (derived from config if None).
        """
        self.config = robot_config
        self.limits = limits or SafetyLimits.from_config(robot_config)
        self._speed_override = 1.0
        self._human_nearby = False
        self._emergency_stop = False

    def check_state(self, state: RobotState) -> SafetyStatus:
        """Check if current state is safe.

        Args:
            state: Current robot state.

        Returns:
            Safety status.
        """
        violations = []
        warnings = []

        # Check joint velocities
        vel_violation = np.abs(state.joint_velocities) > self.limits.max_joint_velocity
        if np.any(vel_violation):
            joints = list(np.where(vel_violation)[0])
            violations.append(f"Joint velocity exceeded on joints {joints}")

        # Check joint torques
        torque_violation = np.abs(state.joint_torques) > self.limits.max_joint_torque
        if np.any(torque_violation):
            joints = list(np.where(torque_violation)[0])
            violations.append(f"Joint torque exceeded on joints {joints}")

        # Check joint limits
        if self.limits.joint_limits_lower is not None:
            lower_violation = state.joint_positions < self.limits.joint_limits_lower
            if np.any(lower_violation):
                joints = list(np.where(lower_violation)[0])
                violations.append(f"Lower joint limit exceeded on joints {joints}")

        if self.limits.joint_limits_upper is not None:
            upper_violation = state.joint_positions > self.limits.joint_limits_upper
            if np.any(upper_violation):
                joints = list(np.where(upper_violation)[0])
                violations.append(f"Upper joint limit exceeded on joints {joints}")

        # Check for approaching limits (warnings)
        if self.limits.joint_limits_upper is not None:
            margin = 0.1  # 0.1 rad margin
            near_upper = state.joint_positions > self.limits.joint_limits_upper - margin
            if np.any(near_upper):
                joints = list(np.where(near_upper)[0])
                warnings.append(f"Approaching upper limit on joints {joints}")

        # Emergency stop check
        if self._emergency_stop:
            violations.append("Emergency stop active")

        # Determine status level
        if violations:
            level = SafetyStatusLevel.ERROR
            is_safe = False
        elif warnings:
            level = SafetyStatusLevel.WARNING
            is_safe = True
        else:
            level = SafetyStatusLevel.OK
            is_safe = True

        return SafetyStatus(
            level=level,
            is_safe=is_safe,
            violations=violations,
            warnings=warnings,
            speed_override=self._speed_override,
        )

    def check_command(self, command: ControlCommand) -> SafetyStatus:
        """Check if command would result in safe state.

        Args:
            command: Control command to check.

        Returns:
            Safety status.
        """
        violations: list[str] = []
        warnings: list[str] = []

        # Check torque limits
        if command.torque_commands is not None:
            torque_violation = (
                np.abs(command.torque_commands) > self.limits.max_joint_torque
            )
            if np.any(torque_violation):
                joints = list(np.where(torque_violation)[0])
                violations.append(f"Command torque exceeds limit on joints {joints}")

        # Check position targets against limits
        if command.position_targets is not None:
            if self.limits.joint_limits_lower is not None:
                lower_violation = (
                    command.position_targets < self.limits.joint_limits_lower
                )
                if np.any(lower_violation):
                    joints = list(np.where(lower_violation)[0])
                    violations.append(f"Position target below limit on joints {joints}")

            if self.limits.joint_limits_upper is not None:
                upper_violation = (
                    command.position_targets > self.limits.joint_limits_upper
                )
                if np.any(upper_violation):
                    joints = list(np.where(upper_violation)[0])
                    violations.append(f"Position target above limit on joints {joints}")

        # Determine status
        if violations:
            level = SafetyStatusLevel.ERROR
            is_safe = False
        elif warnings:
            level = SafetyStatusLevel.WARNING
            is_safe = True
        else:
            level = SafetyStatusLevel.OK
            is_safe = True

        return SafetyStatus(
            level=level,
            is_safe=is_safe,
            violations=violations,
            warnings=warnings,
            speed_override=self._speed_override,
        )

    def compute_safe_command(
        self,
        desired: ControlCommand,
        state: RobotState,
    ) -> ControlCommand:
        """Modify command to ensure safety.

        Args:
            desired: Desired control command.
            state: Current robot state.

        Returns:
            Safe control command.
        """
        from src.deployment.realtime import ControlCommand

        # Start with desired command
        safe_command = ControlCommand(
            timestamp=desired.timestamp,
            mode=desired.mode,
            position_targets=(
                desired.position_targets.copy()
                if desired.position_targets is not None
                else None
            ),
            velocity_targets=(
                desired.velocity_targets.copy()
                if desired.velocity_targets is not None
                else None
            ),
            torque_commands=(
                desired.torque_commands.copy()
                if desired.torque_commands is not None
                else None
            ),
            feedforward_torque=(
                desired.feedforward_torque.copy()
                if desired.feedforward_torque is not None
                else None
            ),
            stiffness=(
                desired.stiffness.copy() if desired.stiffness is not None else None
            ),
            damping=desired.damping.copy() if desired.damping is not None else None,
        )

        # Apply speed override
        if self._speed_override < 1.0:
            if safe_command.velocity_targets is not None:
                safe_command.velocity_targets *= self._speed_override
            if safe_command.torque_commands is not None:
                safe_command.torque_commands *= self._speed_override

        # Clip torque commands
        if safe_command.torque_commands is not None:
            safe_command.torque_commands = np.clip(
                safe_command.torque_commands,
                -self.limits.max_joint_torque,
                self.limits.max_joint_torque,
            )

        # Clip position targets
        if safe_command.position_targets is not None:
            if self.limits.joint_limits_lower is not None:
                safe_command.position_targets = np.maximum(  # type: ignore[arg-type]
                    safe_command.position_targets,
                    self.limits.joint_limits_lower,
                )
            if self.limits.joint_limits_upper is not None:
                safe_command.position_targets = np.minimum(
                    safe_command.position_targets,
                    self.limits.joint_limits_upper,
                )

        return safe_command

    def get_stopping_distance(
        self,
        state: RobotState,
        body: str,
    ) -> float:
        """Compute minimum stopping distance for a body.

        Args:
            state: Current robot state.
            body: Name of the body.

        Returns:
            Minimum stopping distance in meters.
        """
        # Simplified: estimate from maximum velocity
        # Full implementation would use dynamics model
        max_vel = float(np.max(np.abs(state.joint_velocities)))
        max_decel = 2.0  # m/s² typical deceleration

        # s = v² / (2a)
        stopping_distance = (max_vel**2) / (2 * max_decel)

        return stopping_distance

    def set_speed_override(self, factor: float) -> None:
        """Set speed reduction factor.

        Args:
            factor: Speed reduction (0.0 to 1.0).
        """
        self._speed_override = max(0.0, min(1.0, factor))

    def set_human_nearby(self, nearby: bool) -> None:
        """Set human proximity flag.

        Args:
            nearby: True if human is within safety distance.
        """
        self._human_nearby = nearby
        if nearby:
            # Reduce speed when human nearby
            self._speed_override = min(self._speed_override, 0.5)

    def trigger_emergency_stop(self) -> None:
        """Trigger emergency stop."""
        self._emergency_stop = True
        self._speed_override = 0.0

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop."""
        self._emergency_stop = False
        self._speed_override = 1.0

    def is_emergency_stopped(self) -> bool:
        """Check if emergency stopped.

        Returns:
            True if emergency stop is active.
        """
        return self._emergency_stop
