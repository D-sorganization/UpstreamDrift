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


def _zone_bounds(
    zone: Any,
) -> tuple[NDArray[np.floating], NDArray[np.floating]] | None:
    """Lower/upper corners of an axis-aligned box zone, or ``None`` if unshaped.

    Accepts ``[xmin, xmax, ymin, ymax, zmin, zmax]`` (shape ``(6,)``) or
    ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`` (shape ``(3, 2)``).
    """
    z = np.asarray(zone, dtype=float)
    if z.shape == (6,):
        return z[0::2], z[1::2]
    if z.shape == (3, 2):
        return z[:, 0], z[:, 1]
    return None


def _append_flagged(mask: Any, message: str, sink: list[str]) -> None:
    """Append ``"<message> [joints]"`` when any joint in ``mask`` is flagged."""
    if np.any(mask):
        joints = list(np.where(mask)[0])
        sink.append(f"{message} {joints}")


def _fmt_point(pt: NDArray[np.floating]) -> str:
    """``[x, y, z]`` with millimetre precision for violation messages."""
    return f"[{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}]"


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
                continue
            bounds = _zone_bounds(zone)
            if bounds is None:
                continue
            lower, upper = bounds
            self._forbidden_obstacles.append(
                Obstacle(
                    name=f"zone_{i}",
                    obstacle_type=ObstacleType.BOX,
                    position=(lower + upper) / 2.0,
                    dimensions=upper - lower,
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

    def _outside_workspace(self, pt: NDArray[np.floating]) -> bool:
        """True when ``pt`` lies outside the configured workspace box."""
        wb = self.limits.workspace_bounds
        if wb is None:
            return False
        return bool(
            pt[0] < wb[0]
            or pt[0] > wb[1]
            or pt[1] < wb[2]
            or pt[1] > wb[3]
            or pt[2] < wb[4]
            or pt[2] > wb[5]
        )

    def _check_joint_bounds(
        self,
        positions: NDArray[np.floating],
        below: str,
        above: str,
        violations: list[str],
    ) -> None:
        """Flag joints outside the configured lower/upper position limits."""
        if self.limits.joint_limits_lower is not None:
            _append_flagged(
                positions < self.limits.joint_limits_lower, below, violations
            )
        if self.limits.joint_limits_upper is not None:
            _append_flagged(
                positions > self.limits.joint_limits_upper, above, violations
            )

    def _check_near_limits(
        self, positions: NDArray[np.floating], warnings: list[str]
    ) -> None:
        """Warn symmetrically when joints approach either position limit."""
        if self.limits.joint_limits_upper is not None:
            _append_flagged(
                positions > self.limits.joint_limits_upper - NEAR_LIMIT_MARGIN_RAD,
                "Approaching upper limit on joints",
                warnings,
            )
        if self.limits.joint_limits_lower is not None:
            _append_flagged(
                positions < self.limits.joint_limits_lower + NEAR_LIMIT_MARGIN_RAD,
                "Approaching lower limit on joints",
                warnings,
            )

    def _check_state_points(
        self,
        fk: Callable[[NDArray[np.floating]], Any],
        positions: NDArray[np.floating],
        violations: list[str],
    ) -> None:
        """Workspace and forbidden-zone checks for the current FK points."""
        try:
            points = self._extract_points(fk(positions))
            wb = self.limits.workspace_bounds
            for name, pt in points:
                if wb is not None and self._outside_workspace(pt):
                    bounds = ", ".join(f"{v:.3f}" for v in wb)
                    violations.append(
                        f"Workspace bounds exceeded for {name}: position "
                        f"{_fmt_point(pt)} outside bounds [{bounds}]"
                    )
                for obs in self._forbidden_obstacles:
                    if obs.get_distance(pt) <= 0:
                        violations.append(
                            f"Forbidden zone violation for {name}: position "
                            f"{_fmt_point(pt)} inside forbidden zone '{obs.name}'"
                        )
        except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
            violations.append(f"FK computation error in check_state: {exc}")

    def _check_command_points(
        self,
        fk: Callable[[NDArray[np.floating]], Any],
        targets: NDArray[np.floating],
        violations: list[str],
    ) -> None:
        """Workspace and forbidden-zone checks for commanded FK points."""
        try:
            points = self._extract_points(fk(targets))
            for name, pt in points:
                if self._outside_workspace(pt):
                    violations.append(
                        f"Command target for {name} outside workspace bounds: "
                        f"position {_fmt_point(pt)}"
                    )
                for obs in self._forbidden_obstacles:
                    if obs.get_distance(pt) <= 0:
                        violations.append(
                            f"Command target for {name} inside forbidden zone "
                            f"'{obs.name}': position {_fmt_point(pt)}"
                        )
        except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
            violations.append(f"FK computation error in check_command: {exc}")

    def _check_cartesian_speed(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
        label: str,
        violations: list[str],
    ) -> None:
        """Flag a Cartesian speed above ``max_cartesian_velocity``."""
        limit = self.limits.max_cartesian_velocity
        cart_vel = self._compute_cartesian_velocity(positions, velocities)
        if cart_vel is None or limit is None:
            return
        cart_speed = float(np.linalg.norm(cart_vel[:3]))
        if cart_speed > limit:
            violations.append(
                f"{label} {cart_speed:.3f} m/s exceeds limit {limit:.3f} m/s"
            )

    def _check_pressure(self, pressure: float, sensor: str, sink: list[str]) -> None:
        """Flag a contact pressure (N/cm²) above ``max_pressure``."""
        limit = self.limits.max_pressure
        if limit is not None and pressure > limit:
            sink.append(
                f"Contact pressure {pressure:.1f} N/cm² exceeds limit "
                f"{limit:.1f} N/cm² on sensor '{sensor}'"
            )

    def _check_force(self, f_mag: float, sensor: str, sink: list[str]) -> None:
        """Contact force, Cartesian force and derived pressure for one sensor."""
        for kind, limit in (
            ("Contact force", self.limits.max_contact_force),
            ("Cartesian force", self.limits.max_cartesian_force),
        ):
            if limit is not None and f_mag > limit:
                sink.append(
                    f"{kind} {f_mag:.1f} N exceeds limit {limit:.1f} N "
                    f"on sensor '{sensor}'"
                )
        area = self._contact_surface_area_cm2
        if area is not None and area > 0:
            self._check_pressure(f_mag / area, sensor, sink)

    def _check_wrenches(self, wrenches: Any, violations: list[str]) -> None:
        """Validate F/T sensor readings and check force and pressure limits."""
        for sensor_name, wrench in (wrenches or {}).items():
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
                self._check_force(
                    float(np.linalg.norm(wrench[:3])), sensor_name, violations
                )
            elif wrench.size == 1:
                self._check_pressure(float(wrench[0]), sensor_name, violations)

    def check_state(self, state: RobotState) -> SafetyStatus:
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

        if vel_valid:
            _append_flagged(
                np.abs(state.joint_velocities) > self.limits.max_joint_velocity,
                "Joint velocity exceeded on joints",
                violations,
            )
        if tau_valid:
            _append_flagged(
                np.abs(state.joint_torques) > self.limits.max_joint_torque,
                "Joint torque exceeded on joints",
                violations,
            )
        if pos_valid:
            self._check_joint_bounds(
                state.joint_positions,
                "Lower joint limit exceeded on joints",
                "Upper joint limit exceeded on joints",
                violations,
            )
            self._check_near_limits(state.joint_positions, warnings)
            if self.fk_callback is not None:
                self._check_state_points(
                    self.fk_callback, state.joint_positions, violations
                )
        if vel_valid and pos_valid:
            self._check_cartesian_speed(
                state.joint_positions,
                state.joint_velocities,
                "Cartesian velocity",
                violations,
            )
        if state.ft_wrenches:
            self._check_wrenches(state.ft_wrenches, violations)

        if self._emergency_stop:
            violations.append("Emergency stop active")

        return self._build_status(violations, warnings)

    def check_command(self, command: ControlCommand) -> SafetyStatus:
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

        if tau_valid and command.torque_commands is not None:
            _append_flagged(
                np.abs(command.torque_commands) > self.limits.max_joint_torque,
                "Command torque exceeds limit on joints",
                violations,
            )
        if vel_valid and command.velocity_targets is not None:
            _append_flagged(
                np.abs(command.velocity_targets) > self.limits.max_joint_velocity,
                "Command velocity exceeds limit on joints",
                violations,
            )
        if pos_valid and command.position_targets is not None:
            self._check_joint_bounds(
                command.position_targets,
                "Position target below limit on joints",
                "Position target above limit on joints",
                violations,
            )
            if self.fk_callback is not None:
                self._check_command_points(
                    self.fk_callback, command.position_targets, violations
                )
        if vel_valid and command.velocity_targets is not None:
            ref_pos = (
                command.position_targets
                if command.position_targets is not None
                else np.zeros(n_joints)
            )
            self._check_cartesian_speed(
                ref_pos,
                command.velocity_targets,
                "Command Cartesian velocity",
                violations,
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
