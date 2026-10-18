"""Safety monitoring for robot operation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.deployment.realtime import ControlCommand, RobotConfig, RobotState
    from src.deployment.safety.collision import Obstacle


def _validate_joint_array(
    arr: np.ndarray | None,
    name: str,
    n_joints: int,
    violations: list[str],
) -> bool:
    """Validate shape and finite values of a joint array.

    Returns:
        True if valid or None; False if shape mismatch or non-finite.
    """
    if arr is None:
        return True
    if not isinstance(arr, np.ndarray) or arr.shape != (n_joints,):
        shape_str = str(getattr(arr, "shape", type(arr)))
        violations.append(
            f"Invalid shape for {name}: expected ({n_joints},), got {shape_str}"
        )
        return False
    if not np.all(np.isfinite(arr)):
        bad_indices = list(np.where(~np.isfinite(arr))[0])
        violations.append(
            f"Non-finite values (NaN/inf) in {name} on joints {bad_indices}"
        )
        return False
    return True


# Margin (rad) from a joint limit at which an approaching-limit WARNING is
# raised. Applied symmetrically to both the upper and lower joint limits.
NEAR_LIMIT_MARGIN_RAD = 0.1


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
    forbidden_zones: list[Any] = field(default_factory=list)
    max_contact_force: float = 150.0  # ISO 10218-1
    max_pressure: float = 110.0  # N/cm² ISO/TS 15066
    joint_limits_lower: NDArray[np.floating] | None = None
    joint_limits_upper: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        """Validate safety limits parameters."""
        if self.max_joint_velocity is None:
            raise ValueError("max_joint_velocity must be provided")
        if self.max_joint_torque is None:
            raise ValueError("max_joint_torque must be provided")

        if not np.all(np.isfinite(self.max_joint_velocity)) or np.any(
            self.max_joint_velocity <= 0
        ):
            raise ValueError("max_joint_velocity must be strictly positive and finite")
        if not np.all(np.isfinite(self.max_joint_torque)) or np.any(
            self.max_joint_torque <= 0
        ):
            raise ValueError("max_joint_torque must be strictly positive and finite")

        if self.max_cartesian_velocity is not None:
            if (
                not math.isfinite(self.max_cartesian_velocity)
                or self.max_cartesian_velocity <= 0
            ):
                raise ValueError(
                    "max_cartesian_velocity must be a positive finite float"
                )
        if self.max_cartesian_force is not None:
            if (
                not math.isfinite(self.max_cartesian_force)
                or self.max_cartesian_force <= 0
            ):
                raise ValueError("max_cartesian_force must be a positive finite float")
        if self.max_contact_force is not None:
            if not math.isfinite(self.max_contact_force) or self.max_contact_force <= 0:
                raise ValueError("max_contact_force must be a positive finite float")
        if self.max_pressure is not None:
            if not math.isfinite(self.max_pressure) or self.max_pressure <= 0:
                raise ValueError("max_pressure must be a positive finite float")

        if self.workspace_bounds is not None:
            wb = np.asarray(self.workspace_bounds, dtype=float)
            if wb.shape == (6,):
                pass
            elif wb.shape == (3, 2):
                wb = wb.flatten()
            else:
                raise ValueError(
                    "workspace_bounds must have shape (6,) [x_min, x_max, y_min, y_max, z_min, z_max] or (3, 2)"
                )
            if not np.all(np.isfinite(wb)):
                raise ValueError("workspace_bounds must contain only finite values")
            if wb[0] >= wb[1] or wb[2] >= wb[3] or wb[4] >= wb[5]:
                raise ValueError(
                    f"workspace_bounds min must be strictly less than max: "
                    f"x=[{wb[0]}, {wb[1]}], y=[{wb[2]}, {wb[3]}], z=[{wb[4]}, {wb[5]}]"
                )
            self.workspace_bounds = wb

        if self.forbidden_zones:
            from src.deployment.safety.collision import Obstacle

            for i, zone in enumerate(self.forbidden_zones):
                if isinstance(zone, Obstacle):
                    continue
                z = np.asarray(zone, dtype=float)
                if z.shape == (6,):
                    pass
                elif z.shape == (3, 2):
                    z = z.flatten()
                else:
                    raise ValueError(
                        f"forbidden_zones[{i}] must have shape (6,) or (3, 2) or be an Obstacle"
                    )
                if not np.all(np.isfinite(z)):
                    raise ValueError(
                        f"forbidden_zones[{i}] must contain only finite values"
                    )
                if z[0] >= z[1] or z[2] >= z[3] or z[4] >= z[5]:
                    raise ValueError(
                        f"forbidden_zones[{i}] min must be strictly less than max"
                    )

        if self.joint_limits_lower is not None and self.joint_limits_upper is not None:
            if np.any(self.joint_limits_lower > self.joint_limits_upper):
                raise ValueError("joint_limits_lower must be <= joint_limits_upper")

    @classmethod
    def from_config(cls, robot_config: RobotConfig) -> SafetyLimits:
        """Create safety limits from robot configuration.

        Args:
            robot_config: Robot configuration.

        Returns:
            Safety limits instance.
        """
        if not (robot_config is not None):
            raise ValueError("robot_config must be provided")
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

    #: Assumed constant angular deceleration (rad/s²) used by
    #: :meth:`get_stopping_distance` for the joint-space braking-angle estimate.
    _STOPPING_DECELERATION_RAD_S2 = 2.0

    def __init__(
        self,
        robot_config: RobotConfig,
        limits: SafetyLimits | None = None,
        fk_callback: Callable[[NDArray[np.floating]], Any] | None = None,
        fk_velocity_callback: Callable[
            [NDArray[np.floating], NDArray[np.floating]], Any
        ]
        | None = None,
        contact_surface_area_cm2: float | None = None,
    ) -> None:
        """Initialize safety monitor.

        Args:
            robot_config: Robot configuration.
            limits: Safety limits (derived from config if None).
            fk_callback: Optional forward kinematics callback mapping joint positions
                to 3D point(s) or dictionary of link positions.
            fk_velocity_callback: Optional callback mapping (positions, velocities)
                to 3D Cartesian linear velocity vector.
            contact_surface_area_cm2: Optional contact surface area for pressure checks (cm²).
        """
        if not (robot_config is not None):
            raise ValueError("robot_config must be provided")
        self.config = robot_config
        self.limits = limits or SafetyLimits.from_config(robot_config)
        self.fk_callback = fk_callback or getattr(robot_config, "fk_callback", None)
        self.fk_velocity_callback = fk_velocity_callback or getattr(
            robot_config, "fk_velocity_callback", None
        )
        self._contact_surface_area_cm2 = contact_surface_area_cm2 or getattr(
            robot_config, "contact_surface_area_cm2", None
        )
        self._speed_override = 1.0
        self._human_nearby = False
        self._emergency_stop = False

        # Validate that configured limits can be enforced
        if self.limits.workspace_bounds is not None and self.fk_callback is None:
            raise ValueError(
                "workspace_bounds limit is configured on SafetyLimits but no fk_callback is provided on RobotConfig or SafetyMonitor"
            )
        if self.limits.forbidden_zones and self.fk_callback is None:
            raise ValueError(
                "forbidden_zones limit is configured on SafetyLimits but no fk_callback is provided on RobotConfig or SafetyMonitor"
            )

        from src.deployment.safety.collision import Obstacle, ObstacleType

        self._forbidden_obstacles: list[Obstacle] = []
        for i, zone in enumerate(self.limits.forbidden_zones):
            if isinstance(zone, Obstacle):
                self._forbidden_obstacles.append(zone)
            else:
                z = np.asarray(zone, dtype=float)
                if z.shape == (6,):
                    center = np.array(
                        [(z[0] + z[1]) / 2.0, (z[2] + z[3]) / 2.0, (z[4] + z[5]) / 2.0]
                    )
                    dims = np.array([z[1] - z[0], z[3] - z[2], z[5] - z[4]])
                    self._forbidden_obstacles.append(
                        Obstacle(
                            name=f"zone_{i}",
                            obstacle_type=ObstacleType.BOX,
                            position=center,
                            dimensions=dims,
                            inflation=0.0,
                        )
                    )
                elif z.shape == (3, 2):
                    center = np.array(
                        [
                            (z[0, 0] + z[0, 1]) / 2.0,
                            (z[1, 0] + z[1, 1]) / 2.0,
                            (z[2, 0] + z[2, 1]) / 2.0,
                        ]
                    )
                    dims = np.array(
                        [
                            z[0, 1] - z[0, 0],
                            z[1, 1] - z[1, 0],
                            z[2, 1] - z[2, 0],
                        ]
                    )
                    self._forbidden_obstacles.append(
                        Obstacle(
                            name=f"zone_{i}",
                            obstacle_type=ObstacleType.BOX,
                            position=center,
                            dimensions=dims,
                            inflation=0.0,
                        )
                    )

    def _build_status(self, violations: list[str], warnings: list[str]) -> SafetyStatus:
        if self._emergency_stop:
            level = SafetyStatusLevel.EMERGENCY
            is_safe = False
        elif violations:
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

    def _extract_points(self, result: Any) -> list[tuple[str, np.ndarray]]:
        """Extract list of named 3D points from FK callback return value."""
        points: list[tuple[str, np.ndarray]] = []
        if isinstance(result, dict):
            for k, v in result.items():
                arr = np.asarray(v, dtype=float)
                if arr.shape == (3,) and np.all(np.isfinite(arr)):
                    points.append((str(k), arr))
        elif isinstance(result, np.ndarray):
            if (
                result.ndim == 1
                and result.shape == (3,)
                and np.all(np.isfinite(result))
            ):
                points.append(("end_effector", result.astype(float)))
            elif (
                result.ndim == 2
                and result.shape[1] == 3
                and np.all(np.isfinite(result))
            ):
                for i, row in enumerate(result):
                    points.append((f"link_{i}", row.astype(float)))
        elif isinstance(result, (list, tuple)):
            if len(result) == 3 and all(
                isinstance(x, (int, float, np.floating)) for x in result
            ):
                arr = np.asarray(result, dtype=float)
                if np.all(np.isfinite(arr)):
                    points.append(("end_effector", arr))
            else:
                for i, item in enumerate(result):
                    arr = np.asarray(item, dtype=float)
                    if arr.shape == (3,) and np.all(np.isfinite(arr)):
                        points.append((f"point_{i}", arr))
        return points

    def _compute_cartesian_velocity(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> np.ndarray | None:
        """Compute Cartesian linear velocity vector from joint state."""
        if self.fk_velocity_callback is not None:
            try:
                res = self.fk_velocity_callback(positions, velocities)
                arr = np.asarray(res, dtype=float)
                if arr.shape == (3,) and np.all(np.isfinite(arr)):
                    return arr
            except (ValueError, TypeError, RuntimeError, ArithmeticError):
                pass
        return None

    def check_state(self, state: RobotState) -> SafetyStatus:  # noqa: C901
        """Check if current state is safe.

        Args:
            state: Current robot state.

        Returns:
            Safety status.
        """
        if not (state is not None):
            raise ValueError("state must be provided")
        violations: list[str] = []
        warnings: list[str] = []
        n_joints = self.config.n_joints

        pos_valid = _validate_joint_array(
            state.joint_positions, "joint_positions", n_joints, violations
        )
        vel_valid = _validate_joint_array(
            state.joint_velocities, "joint_velocities", n_joints, violations
        )
        tau_valid = _validate_joint_array(
            state.joint_torques, "joint_torques", n_joints, violations
        )

        # Check joint velocities
        if vel_valid:
            vel_violation = (
                np.abs(state.joint_velocities) > self.limits.max_joint_velocity
            )
            if np.any(vel_violation):
                joints = list(np.where(vel_violation)[0])
                violations.append(f"Joint velocity exceeded on joints {joints}")

        # Check joint torques
        if tau_valid:
            torque_violation = (
                np.abs(state.joint_torques) > self.limits.max_joint_torque
            )
            if np.any(torque_violation):
                joints = list(np.where(torque_violation)[0])
                violations.append(f"Joint torque exceeded on joints {joints}")

        # Check joint limits
        if pos_valid:
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

            # Check for approaching limits (warnings) — symmetric on both bounds.
            if self.limits.joint_limits_upper is not None:
                near_upper = (
                    state.joint_positions
                    > self.limits.joint_limits_upper - NEAR_LIMIT_MARGIN_RAD
                )
                if np.any(near_upper):
                    joints = list(np.where(near_upper)[0])
                    warnings.append(f"Approaching upper limit on joints {joints}")

            if self.limits.joint_limits_lower is not None:
                near_lower = (
                    state.joint_positions
                    < self.limits.joint_limits_lower + NEAR_LIMIT_MARGIN_RAD
                )
                if np.any(near_lower):
                    joints = list(np.where(near_lower)[0])
                    warnings.append(f"Approaching lower limit on joints {joints}")

        # Check workspace bounds and forbidden zones via FK
        if pos_valid and self.fk_callback is not None:
            try:
                fk_res = self.fk_callback(state.joint_positions)
                points = self._extract_points(fk_res)
                for name, pt in points:
                    if self.limits.workspace_bounds is not None:
                        wb = self.limits.workspace_bounds
                        if (
                            pt[0] < wb[0]
                            or pt[0] > wb[1]
                            or pt[1] < wb[2]
                            or pt[1] > wb[3]
                            or pt[2] < wb[4]
                            or pt[2] > wb[5]
                        ):
                            violations.append(
                                f"Workspace bounds exceeded for {name}: position "
                                f"[{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}] outside bounds "
                                f"[{wb[0]:.3f}, {wb[1]:.3f}, {wb[2]:.3f}, {wb[3]:.3f}, {wb[4]:.3f}, {wb[5]:.3f}]"
                            )
                    for obs in self._forbidden_obstacles:
                        if obs.get_distance(pt) <= 0:
                            violations.append(
                                f"Forbidden zone violation for {name}: position "
                                f"[{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}] inside forbidden zone '{obs.name}'"
                            )
            except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
                violations.append(f"FK computation error in check_state: {exc}")

        # Check Cartesian velocity
        if vel_valid and pos_valid:
            cart_vel = self._compute_cartesian_velocity(
                state.joint_positions, state.joint_velocities
            )
            if cart_vel is not None and self.limits.max_cartesian_velocity is not None:
                cart_speed = float(np.linalg.norm(cart_vel[:3]))
                if cart_speed > self.limits.max_cartesian_velocity:
                    violations.append(
                        f"Cartesian velocity {cart_speed:.3f} m/s exceeds limit {self.limits.max_cartesian_velocity:.3f} m/s"
                    )

        # Check contact force, Cartesian force, and pressure from F/T sensors
        if state.ft_wrenches:
            for sensor_name, wrench in state.ft_wrenches.items():
                if wrench is None:
                    continue
                if not isinstance(wrench, np.ndarray):
                    violations.append(
                        f"Invalid type for ft_wrenches on sensor '{sensor_name}'"
                    )
                    continue
                if not np.all(np.isfinite(wrench)):
                    violations.append(
                        f"Non-finite values in ft_wrenches on sensor '{sensor_name}'"
                    )
                    continue

                if wrench.size >= 3:
                    f_mag = float(np.linalg.norm(wrench[:3]))
                    if (
                        self.limits.max_contact_force is not None
                        and f_mag > self.limits.max_contact_force
                    ):
                        violations.append(
                            f"Contact force {f_mag:.1f} N exceeds limit {self.limits.max_contact_force:.1f} N on sensor '{sensor_name}'"
                        )
                    if (
                        self.limits.max_cartesian_force is not None
                        and f_mag > self.limits.max_cartesian_force
                    ):
                        violations.append(
                            f"Cartesian force {f_mag:.1f} N exceeds limit {self.limits.max_cartesian_force:.1f} N on sensor '{sensor_name}'"
                        )
                    area = self._contact_surface_area_cm2
                    if (
                        area is not None
                        and area > 0
                        and self.limits.max_pressure is not None
                    ):
                        pressure = f_mag / area
                        if pressure > self.limits.max_pressure:
                            violations.append(
                                f"Contact pressure {pressure:.1f} N/cm² exceeds limit {self.limits.max_pressure:.1f} N/cm² on sensor '{sensor_name}'"
                            )
                elif wrench.size == 1:
                    p_val = float(wrench[0])
                    if (
                        self.limits.max_pressure is not None
                        and p_val > self.limits.max_pressure
                    ):
                        violations.append(
                            f"Contact pressure {p_val:.1f} N/cm² exceeds limit {self.limits.max_pressure:.1f} N/cm² on sensor '{sensor_name}'"
                        )

        # Emergency stop check
        if self._emergency_stop:
            violations.append("Emergency stop active")

        return self._build_status(violations, warnings)

    def check_command(self, command: ControlCommand) -> SafetyStatus:  # noqa: C901
        """Check if command would result in safe state.

        Args:
            command: Control command to check.

        Returns:
            Safety status.
        """
        if not (command is not None):
            raise ValueError("command must be provided")
        violations: list[str] = []
        warnings: list[str] = []
        n_joints = self.config.n_joints

        pos_valid = _validate_joint_array(
            command.position_targets, "position_targets", n_joints, violations
        )
        vel_valid = _validate_joint_array(
            command.velocity_targets, "velocity_targets", n_joints, violations
        )
        tau_valid = _validate_joint_array(
            command.torque_commands, "torque_commands", n_joints, violations
        )
        _validate_joint_array(
            command.feedforward_torque, "feedforward_torque", n_joints, violations
        )

        # Check torque limits
        if tau_valid and command.torque_commands is not None:
            torque_violation = (
                np.abs(command.torque_commands) > self.limits.max_joint_torque
            )
            if np.any(torque_violation):
                joints = list(np.where(torque_violation)[0])
                violations.append(f"Command torque exceeds limit on joints {joints}")

        # Check velocity limits
        if vel_valid and command.velocity_targets is not None:
            velocity_violation = (
                np.abs(command.velocity_targets) > self.limits.max_joint_velocity
            )
            if np.any(velocity_violation):
                joints = list(np.where(velocity_violation)[0])
                violations.append(f"Command velocity exceeds limit on joints {joints}")

        # Check position targets against limits
        if pos_valid and command.position_targets is not None:
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

        # Check workspace bounds and forbidden zones for command position targets
        if (
            pos_valid
            and command.position_targets is not None
            and self.fk_callback is not None
        ):
            try:
                fk_res = self.fk_callback(command.position_targets)
                points = self._extract_points(fk_res)
                for name, pt in points:
                    if self.limits.workspace_bounds is not None:
                        wb = self.limits.workspace_bounds
                        if (
                            pt[0] < wb[0]
                            or pt[0] > wb[1]
                            or pt[1] < wb[2]
                            or pt[1] > wb[3]
                            or pt[2] < wb[4]
                            or pt[2] > wb[5]
                        ):
                            violations.append(
                                f"Command target for {name} outside workspace bounds: position "
                                f"[{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}]"
                            )
                    for obs in self._forbidden_obstacles:
                        if obs.get_distance(pt) <= 0:
                            violations.append(
                                f"Command target for {name} inside forbidden zone '{obs.name}': position "
                                f"[{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}]"
                            )
            except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
                violations.append(f"FK computation error in check_command: {exc}")

        # Check command Cartesian velocity
        if vel_valid and command.velocity_targets is not None:
            ref_pos = (
                command.position_targets
                if command.position_targets is not None
                else np.zeros(n_joints)
            )
            cart_vel = self._compute_cartesian_velocity(
                ref_pos, command.velocity_targets
            )
            if cart_vel is not None and self.limits.max_cartesian_velocity is not None:
                cart_speed = float(np.linalg.norm(cart_vel[:3]))
                if cart_speed > self.limits.max_cartesian_velocity:
                    violations.append(
                        f"Command Cartesian velocity {cart_speed:.3f} m/s exceeds limit {self.limits.max_cartesian_velocity:.3f} m/s"
                    )

        if self._emergency_stop:
            violations.append("Emergency stop active")

        return self._build_status(violations, warnings)

    def _check_invalid_command_or_state(
        self,
        state: RobotState,
        desired: ControlCommand,
    ) -> bool:
        """Check if desired command or state has non-finite entries or shape mismatch."""
        n_joints = self.config.n_joints
        return (
            not _validate_joint_array(
                state.joint_positions, "joint_positions", n_joints, []
            )
            or not _validate_joint_array(
                state.joint_velocities, "joint_velocities", n_joints, []
            )
            or not _validate_joint_array(
                state.joint_torques, "joint_torques", n_joints, []
            )
            or not _validate_joint_array(
                desired.position_targets, "position_targets", n_joints, []
            )
            or not _validate_joint_array(
                desired.velocity_targets, "velocity_targets", n_joints, []
            )
            or not _validate_joint_array(
                desired.torque_commands, "torque_commands", n_joints, []
            )
            or not _validate_joint_array(
                desired.feedforward_torque, "feedforward_torque", n_joints, []
            )
        )

    def _apply_fallback_or_estop(
        self,
        safe_command: ControlCommand,
        state: RobotState,
    ) -> None:
        """Zero out velocities/torques and hold position on E-stop or invalid input."""
        n_joints = self.config.n_joints
        if safe_command.position_targets is not None:
            if (
                state.joint_positions is not None
                and isinstance(state.joint_positions, np.ndarray)
                and state.joint_positions.shape == (n_joints,)
                and np.all(np.isfinite(state.joint_positions))
            ):
                safe_command.position_targets = state.joint_positions.copy()
            else:
                safe_command.position_targets = np.zeros(n_joints)
        if safe_command.velocity_targets is not None:
            safe_command.velocity_targets = np.zeros(n_joints)
        if safe_command.torque_commands is not None:
            safe_command.torque_commands = np.zeros(n_joints)
        if safe_command.feedforward_torque is not None:
            safe_command.feedforward_torque = np.zeros(n_joints)

    def _clip_command_limits(self, safe_command: ControlCommand) -> None:
        """Clip velocity, torque, and position targets to safety limits."""
        if safe_command.velocity_targets is not None:
            safe_command.velocity_targets = np.clip(
                safe_command.velocity_targets,
                -self.limits.max_joint_velocity,
                self.limits.max_joint_velocity,
            )
        if safe_command.torque_commands is not None:
            safe_command.torque_commands = np.clip(
                safe_command.torque_commands,
                -self.limits.max_joint_torque,
                self.limits.max_joint_torque,
            )
        if safe_command.position_targets is not None:
            pos_targets = safe_command.position_targets
            if self.limits.joint_limits_lower is not None:
                pos_targets = np.maximum(pos_targets, self.limits.joint_limits_lower)
            if self.limits.joint_limits_upper is not None:
                pos_targets = np.minimum(pos_targets, self.limits.joint_limits_upper)
            safe_command.position_targets = pos_targets

    def _sanitize_finite_command_arrays(self, safe_command: ControlCommand) -> None:
        """Ensure all output command arrays are strictly finite."""
        for attr in (
            "position_targets",
            "velocity_targets",
            "torque_commands",
            "feedforward_torque",
        ):
            val = getattr(safe_command, attr)
            if val is not None and not np.all(np.isfinite(val)):
                setattr(
                    safe_command,
                    attr,
                    np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0),
                )

    def compute_safe_command(
        self,
        desired: ControlCommand,
        state: RobotState,
    ) -> ControlCommand:
        """Modify command to ensure safety."""
        if desired is None:
            raise ValueError("desired must be provided")
        if state is None:
            raise ValueError("state must be provided")
        from src.deployment.realtime import ControlCommand

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
            damping=(desired.damping.copy() if desired.damping is not None else None),
        )

        if self._speed_override < 1.0:
            if safe_command.velocity_targets is not None:
                safe_command.velocity_targets *= self._speed_override
            if safe_command.torque_commands is not None:
                safe_command.torque_commands *= self._speed_override

        has_invalid = self._check_invalid_command_or_state(state, desired)
        cmd_status = self.check_command(desired)
        has_zone_or_ws_violation = any(
            "workspace bounds" in v.lower() or "forbidden zone" in v.lower()
            for v in cmd_status.violations
        )
        if self._emergency_stop or has_invalid or has_zone_or_ws_violation:
            self._apply_fallback_or_estop(safe_command, state)

        self._clip_command_limits(safe_command)
        self._sanitize_finite_command_arrays(safe_command)
        return safe_command

    def get_stopping_distance(
        self,
        state: RobotState,
    ) -> float:
        """Estimate the worst-case joint-space stopping angle.

        This is a conservative, body-agnostic heuristic. A true per-body
        Cartesian stopping *distance* would require the robot's forward
        kinematics / Jacobian to map joint velocities (rad/s) to end-effector
        velocity (m/s), which this monitor does not hold. The previous
        signature took a ``body`` name that was never read (every body got the
        same answer) and mixed rad/s with m/s² to claim "meters"; both were
        misleading, so the parameter was removed (issue #7740).

        The returned value is the braking *angle* of the fastest joint under a
        constant angular deceleration, ``s = ω² / (2·α)``, expressed in
        radians. Callers that need a Cartesian clearance must multiply by the
        relevant link length using their own kinematics model.

        Args:
            state: Current robot state.

        Returns:
            Worst-case joint braking angle in radians (>= 0).
        """
        if not (state is not None):
            raise ValueError("state must be provided")
        if not np.all(np.isfinite(state.joint_velocities)):
            raise ValueError("joint_velocities must be finite")
        max_joint_vel = float(np.max(np.abs(state.joint_velocities)))  # rad/s
        max_decel = self._STOPPING_DECELERATION_RAD_S2  # rad/s²

        # s = ω² / (2·α) — units: (rad/s)² / (rad/s²) = rad.
        return (max_joint_vel**2) / (2 * max_decel)

    def set_speed_override(self, factor: float) -> None:
        """Set speed reduction factor.

        Args:
            factor: Speed reduction (0.0 to 1.0).
        """
        if not math.isfinite(factor):
            raise ValueError(
                "Speed override factor must be a finite float between 0.0 and 1.0"
            )
        if self._emergency_stop:
            self._speed_override = 0.0
            return
        self._speed_override = max(0.0, min(1.0, factor))

    def set_human_nearby(self, nearby: bool) -> None:
        """Set human proximity flag.

        Args:
            nearby: True if human is within safety distance.
        """
        if not (nearby is not None):
            raise ValueError("nearby must be provided")
        self._human_nearby = nearby
        if nearby:
            # Reduce speed when human nearby
            self._speed_override = min(self._speed_override, 0.5)

    def trigger_emergency_stop(self) -> None:
        """Trigger emergency stop."""
        self._emergency_stop = True
        self._speed_override = 0.0

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop.

        Restores motion but preserves the human-proximity speed derate: if a
        human is still nearby (issue #7691) the speed override is capped at the
        same 0.5 factor applied by :meth:`set_human_nearby`, rather than
        jumping back to full speed.
        """
        self._emergency_stop = False
        if self._human_nearby:
            self._speed_override = 0.5
        else:
            self._speed_override = 1.0

    def is_emergency_stopped(self) -> bool:
        """Check if emergency stopped.

        Returns:
            True if emergency stop is active.
        """
        return self._emergency_stop
