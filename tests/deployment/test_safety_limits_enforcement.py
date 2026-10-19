"""Tests for SafetyLimits enforcement and SafetyMonitor integration (issue #12974)."""

from __future__ import annotations

import numpy as np
import pytest

from src.deployment.realtime import (
    ControlCommand,
    ControlMode,
    RobotConfig,
    RobotState,
)
from src.deployment.safety import (
    Obstacle,
    ObstacleType,
    SafetyLimits,
    SafetyMonitor,
    SafetyStatusLevel,
)

pytestmark = pytest.mark.unit


def _fk_linear_3link(q: np.ndarray) -> np.ndarray:
    """Simple 3-link planar FK returning [x, y, z] of end-effector."""
    # link lengths: 0.5, 0.4, 0.3 along x when angles are 0
    x = (
        0.5 * np.cos(q[0])
        + 0.4 * np.cos(q[0] + q[1])
        + 0.3 * np.cos(q[0] + q[1] + q[2])
    )
    y = (
        0.5 * np.sin(q[0])
        + 0.4 * np.sin(q[0] + q[1])
        + 0.3 * np.sin(q[0] + q[1] + q[2])
    )
    z = 0.1
    return np.array([x, y, z])


def _fk_velocity_3link(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Simple 3-link Jacobian-based end-effector linear velocity."""
    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c01 = np.cos(q[0] + q[1])
    s01 = np.sin(q[0] + q[1])
    c012 = np.cos(q[0] + q[1] + q[2])
    s012 = np.sin(q[0] + q[1] + q[2])

    j00 = -0.5 * s0 - 0.4 * s01 - 0.3 * s012
    j01 = -0.4 * s01 - 0.3 * s012
    j02 = -0.3 * s012

    j10 = 0.5 * c0 + 0.4 * c01 + 0.3 * c012
    j11 = 0.4 * c01 + 0.3 * c012
    j12 = 0.3 * c012

    vx = j00 * qd[0] + j01 * qd[1] + j02 * qd[2]
    vy = j10 * qd[0] + j11 * qd[1] + j12 * qd[2]
    vz = 0.0
    return np.array([vx, vy, vz])


class TestSafetyMonitorEmergencyLevel:
    """Verify SafetyStatusLevel.EMERGENCY is produced when emergency stop is active."""

    def test_emergency_stop_sets_emergency_level_in_check_state(self) -> None:
        config = RobotConfig(name="test", n_joints=3)
        monitor = SafetyMonitor(config)
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
        )

        monitor.trigger_emergency_stop()
        status = monitor.check_state(state)

        assert not status.is_safe
        assert status.level == SafetyStatusLevel.EMERGENCY
        assert any("Emergency stop" in v for v in status.violations)

    def test_emergency_stop_sets_emergency_level_in_check_command(self) -> None:
        config = RobotConfig(name="test", n_joints=3)
        monitor = SafetyMonitor(config)
        cmd = ControlCommand.torque_command(0.0, np.zeros(3))

        monitor.trigger_emergency_stop()
        status = monitor.check_command(cmd)

        assert not status.is_safe
        assert status.level == SafetyStatusLevel.EMERGENCY
        assert any("Emergency stop" in v for v in status.violations)


class TestSafetyLimitsValidationAndEnforceability:
    """Verify setting unenforceable limits raises at construction."""

    def test_workspace_bounds_without_fk_raises(self) -> None:
        config = RobotConfig(name="test", n_joints=3)
        limits = SafetyLimits(
            max_joint_velocity=np.ones(3) * 2.0,
            max_joint_torque=np.ones(3) * 50.0,
            workspace_bounds=np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 2.0]),
        )
        with pytest.raises(ValueError, match="workspace_bounds"):
            SafetyMonitor(config, limits)

    def test_forbidden_zones_without_fk_raises(self) -> None:
        config = RobotConfig(name="test", n_joints=3)
        limits = SafetyLimits(
            max_joint_velocity=np.ones(3) * 2.0,
            max_joint_torque=np.ones(3) * 50.0,
            forbidden_zones=[np.array([0.2, 0.4, 0.2, 0.4, 0.0, 1.0])],
        )
        with pytest.raises(ValueError, match="forbidden_zones"):
            SafetyMonitor(config, limits)

    def test_workspace_bounds_min_greater_than_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min must be"):
            SafetyLimits(
                max_joint_velocity=np.ones(3) * 2.0,
                max_joint_torque=np.ones(3) * 50.0,
                workspace_bounds=np.array([1.0, -1.0, -1.0, 1.0, 0.0, 2.0]),
            )

    def test_forbidden_zone_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            SafetyLimits(
                max_joint_velocity=np.ones(3) * 2.0,
                max_joint_torque=np.ones(3) * 50.0,
                forbidden_zones=[np.array([0.1, 0.2])],
            )

    @pytest.mark.parametrize(
        ("field_name", "bad_val"),
        [
            ("max_cartesian_velocity", -1.0),
            ("max_cartesian_velocity", float("nan")),
            ("max_cartesian_force", 0.0),
            ("max_contact_force", -5.0),
            ("max_pressure", float("inf")),
        ],
    )
    def test_non_positive_or_non_finite_limits_raise(
        self, field_name: str, bad_val: float
    ) -> None:
        kwargs: dict[str, object] = {
            "max_joint_velocity": np.ones(3) * 2.0,
            "max_joint_torque": np.ones(3) * 50.0,
            field_name: bad_val,
        }
        with pytest.raises(ValueError, match=field_name):
            SafetyLimits(**kwargs)  # type: ignore[arg-type]


class TestWorkspaceBoundsEnforcement:
    """Verify workspace bounds are enforced in check_state and check_command."""

    @pytest.fixture
    def monitor_with_fk(self) -> SafetyMonitor:
        config = RobotConfig(
            name="test",
            n_joints=3,
            fk_callback=_fk_linear_3link,
        )
        limits = SafetyLimits(
            max_joint_velocity=np.ones(3) * 5.0,
            max_joint_torque=np.ones(3) * 50.0,
            workspace_bounds=np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 0.5]),
        )
        return SafetyMonitor(config, limits)

    def test_state_inside_workspace_is_safe(
        self, monitor_with_fk: SafetyMonitor
    ) -> None:
        # q = [0, 0, 0] gives end-effector at [1.2, 0, 0.1], which is outside x=[-1, 1]!
        # q = [pi/2, 0, 0] gives end-effector at [0, 1.2, 0.1], also outside y=[-1, 1]!
        # q = [pi/2, pi/2, 0] -> link 1 along y (0.5), link 2 along -x (0.4), link 3 along -x (0.3)
        # x = -0.7, y = 0.5, z = 0.1 -> inside [-1, 1, -1, 1, 0, 0.5]!
        q = np.array([np.pi / 2, np.pi / 2, 0.0])
        state = RobotState(
            timestamp=0.0,
            joint_positions=q,
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
        )
        status = monitor_with_fk.check_state(state)
        assert status.is_safe
        assert status.level == SafetyStatusLevel.OK

    def test_state_outside_workspace_violates(
        self, monitor_with_fk: SafetyMonitor
    ) -> None:
        # q = [0, 0, 0] gives end-effector at [1.2, 0, 0.1], outside x_max=1.0
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
        )
        status = monitor_with_fk.check_state(state)
        assert not status.is_safe
        assert status.level == SafetyStatusLevel.ERROR
        assert any("Workspace bounds" in v for v in status.violations)

    def test_command_outside_workspace_violates(
        self, monitor_with_fk: SafetyMonitor
    ) -> None:
        cmd = ControlCommand.position_command(0.0, np.zeros(3))
        status = monitor_with_fk.check_command(cmd)
        assert not status.is_safe
        assert status.level == SafetyStatusLevel.ERROR
        assert any("workspace bounds" in v for v in status.violations)


class TestForbiddenZonesEnforcement:
    """Verify forbidden zones are enforced in check_state, check_command, and compute_safe_command."""

    @pytest.fixture
    def monitor_with_zone(self) -> SafetyMonitor:
        config = RobotConfig(name="test", n_joints=3)
        limits = SafetyLimits(
            max_joint_velocity=np.ones(3) * 5.0,
            max_joint_torque=np.ones(3) * 50.0,
            forbidden_zones=[
                # Box covering [0.5, 1.5] x [-0.5, 0.5] x [0.0, 0.5]
                np.array([0.5, 1.5, -0.5, 0.5, 0.0, 0.5]),
            ],
        )
        return SafetyMonitor(config, limits, fk_callback=_fk_linear_3link)

    def test_state_inside_forbidden_zone_violates(
        self, monitor_with_zone: SafetyMonitor
    ) -> None:
        # q = [0, 0, 0] -> EE at [1.2, 0.0, 0.1] (inside forbidden box!)
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
        )
        status = monitor_with_zone.check_state(state)
        assert not status.is_safe
        assert status.level == SafetyStatusLevel.ERROR
        assert any("Forbidden zone" in v for v in status.violations)

    def test_state_outside_forbidden_zone_is_safe(
        self, monitor_with_zone: SafetyMonitor
    ) -> None:
        # q = [pi, 0, 0] -> EE at [-1.2, 0.0, 0.1] (outside box)
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.array([np.pi, 0.0, 0.0]),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
        )
        status = monitor_with_zone.check_state(state)
        assert status.is_safe

    def test_command_inside_forbidden_zone_violates(
        self, monitor_with_zone: SafetyMonitor
    ) -> None:
        cmd = ControlCommand.position_command(0.0, np.zeros(3))
        status = monitor_with_zone.check_command(cmd)
        assert not status.is_safe
        assert any("forbidden zone" in v for v in status.violations)

    def test_compute_safe_command_freezes_position_on_forbidden_target(
        self, monitor_with_zone: SafetyMonitor
    ) -> None:
        safe_pos = np.array([np.pi, 0.0, 0.0])
        state = RobotState(
            timestamp=0.0,
            joint_positions=safe_pos,
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
        )
        # Desired command aims directly into forbidden zone
        cmd = ControlCommand.position_command(0.0, np.zeros(3))
        safe_cmd = monitor_with_zone.compute_safe_command(cmd, state)

        # Must hold current safe position instead of commanding forbidden target
        assert safe_cmd.position_targets is not None
        np.testing.assert_allclose(safe_cmd.position_targets, safe_pos)

    def test_obstacle_instance_in_forbidden_zones(self) -> None:
        """Verify Obstacle instance from collision.py works as forbidden zone."""
        obs = Obstacle(
            name="table",
            obstacle_type=ObstacleType.BOX,
            position=np.array([1.0, 0.0, 0.1]),
            dimensions=np.array([0.5, 0.5, 0.2]),
            inflation=0.0,
        )
        config = RobotConfig(name="test", n_joints=3)
        limits = SafetyLimits(
            max_joint_velocity=np.ones(3) * 5.0,
            max_joint_torque=np.ones(3) * 50.0,
            forbidden_zones=[obs],
        )
        monitor = SafetyMonitor(config, limits, fk_callback=_fk_linear_3link)
        # EE at [1.0, 0, 0.1]:
        # Solve q roughly: q0=0.2, etc. Or let's test directly with mock callback returning obstacle center
        monitor.fk_callback = lambda _q: np.array([1.0, 0.0, 0.1])
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
        )
        status = monitor.check_state(state)
        assert not status.is_safe
        assert any("table" in v for v in status.violations)


class TestContactForceAndPressureEnforcement:
    """Verify max_contact_force, max_cartesian_force, and max_pressure enforcement."""

    def test_contact_force_limit_enforced(self) -> None:
        config = RobotConfig(name="test", n_joints=3)
        limits = SafetyLimits(
            max_joint_velocity=np.ones(3) * 2.0,
            max_joint_torque=np.ones(3) * 50.0,
            max_contact_force=100.0,
        )
        monitor = SafetyMonitor(config, limits)

        # 80 N force: safe
        safe_state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
            ft_wrenches={"wrist": np.array([0.0, 0.0, 80.0, 0.0, 0.0, 0.0])},
        )
        assert monitor.check_state(safe_state).is_safe

        # 120 N force: violation
        unsafe_state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
            ft_wrenches={"wrist": np.array([0.0, 0.0, 120.0, 0.0, 0.0, 0.0])},
        )
        status = monitor.check_state(unsafe_state)
        assert not status.is_safe
        assert any("Contact force" in v and "wrist" in v for v in status.violations)

    def test_cartesian_force_limit_enforced(self) -> None:
        config = RobotConfig(name="test", n_joints=3)
        limits = SafetyLimits(
            max_joint_velocity=np.ones(3) * 2.0,
            max_joint_torque=np.ones(3) * 50.0,
            max_cartesian_force=75.0,
            max_contact_force=200.0,
        )
        monitor = SafetyMonitor(config, limits)

        unsafe_state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
            ft_wrenches={
                "ee": np.array([50.0, 60.0, 0.0, 0.0, 0.0, 0.0])
            },  # norm ~ 78.1 N
        )
        status = monitor.check_state(unsafe_state)
        assert not status.is_safe
        assert any("Cartesian force" in v for v in status.violations)

    def test_contact_pressure_limit_enforced_with_surface_area(self) -> None:
        # 10 cm^2 area, 1200 N force -> 120 N/cm^2 > limit 110 N/cm^2
        config = RobotConfig(name="test", n_joints=3, contact_surface_area_cm2=10.0)
        limits = SafetyLimits(
            max_joint_velocity=np.ones(3) * 2.0,
            max_joint_torque=np.ones(3) * 50.0,
            max_contact_force=2000.0,
            max_pressure=110.0,
        )
        monitor = SafetyMonitor(config, limits)

        unsafe_state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
            ft_wrenches={"contact": np.array([0.0, 0.0, 1200.0, 0.0, 0.0, 0.0])},
        )
        status = monitor.check_state(unsafe_state)
        assert not status.is_safe
        assert any("Contact pressure" in v for v in status.violations)


class TestCartesianVelocityEnforcement:
    """Verify max_cartesian_velocity enforcement via velocity callback."""

    def test_cartesian_velocity_limit_enforced_in_check_state(self) -> None:
        config = RobotConfig(name="test", n_joints=3)
        limits = SafetyLimits(
            max_joint_velocity=np.ones(3) * 10.0,
            max_joint_torque=np.ones(3) * 50.0,
            max_cartesian_velocity=0.25,
        )
        monitor = SafetyMonitor(
            config,
            limits,
            fk_velocity_callback=_fk_velocity_3link,
        )

        # Small velocity -> Cartesian speed < 0.25 m/s
        safe_state = RobotState(
            timestamp=0.0,
            joint_positions=np.array([0.0, 0.0, 0.0]),
            joint_velocities=np.array([0.1, 0.0, 0.0]),
            joint_torques=np.zeros(3),
        )
        assert monitor.check_state(safe_state).is_safe

        # Large velocity -> Cartesian speed > 0.25 m/s
        unsafe_state = RobotState(
            timestamp=0.0,
            joint_positions=np.array([0.0, 0.0, 0.0]),
            joint_velocities=np.array([1.0, 0.0, 0.0]),  # yields vy = 1.2 m/s > 0.25
            joint_torques=np.zeros(3),
        )
        status = monitor.check_state(unsafe_state)
        assert not status.is_safe
        assert any("Cartesian velocity" in v for v in status.violations)

    def test_cartesian_velocity_limit_enforced_in_check_command(self) -> None:
        config = RobotConfig(name="test", n_joints=3)
        limits = SafetyLimits(
            max_joint_velocity=np.ones(3) * 10.0,
            max_joint_torque=np.ones(3) * 50.0,
            max_cartesian_velocity=0.25,
        )
        monitor = SafetyMonitor(
            config,
            limits,
            fk_velocity_callback=_fk_velocity_3link,
        )

        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.VELOCITY,
            position_targets=np.zeros(3),
            velocity_targets=np.array([1.0, 0.0, 0.0]),
        )
        status = monitor.check_command(cmd)
        assert not status.is_safe
        assert any("Cartesian velocity" in v for v in status.violations)
