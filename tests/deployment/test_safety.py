"""Tests for safety module."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestSafetyLimits:
    """Tests for SafetyLimits."""

    def test_safety_limits_creation(self) -> None:
        """Test creating safety limits."""
        from src.deployment.safety import SafetyLimits

        limits = SafetyLimits(
            max_joint_velocity=np.ones(7) * 2.0,
            max_joint_torque=np.ones(7) * 50.0,
        )

        assert limits.max_cartesian_velocity == 0.25
        assert limits.max_contact_force == 150.0

    def test_safety_limits_from_config(self) -> None:
        """Test creating limits from robot config."""
        from src.deployment.realtime import RobotConfig
        from src.deployment.safety import SafetyLimits

        config = RobotConfig(
            name="test",
            n_joints=7,
            velocity_limits=np.ones(7) * 3.0,
            torque_limits=np.ones(7) * 100.0,
        )

        limits = SafetyLimits.from_config(config)

        np.testing.assert_array_equal(limits.max_joint_velocity, config.velocity_limits)
        np.testing.assert_array_equal(limits.max_joint_torque, config.torque_limits)


class TestSafetyMonitor:
    """Tests for SafetyMonitor."""

    def test_monitor_creation(self) -> None:
        """Test creating safety monitor."""
        from src.deployment.realtime import RobotConfig
        from src.deployment.safety import SafetyMonitor

        config = RobotConfig(name="test", n_joints=7)
        monitor = SafetyMonitor(config)

        assert not monitor.is_emergency_stopped()

    def test_check_safe_state(self) -> None:
        """Test checking a safe state."""
        from src.deployment.realtime import RobotConfig, RobotState
        from src.deployment.safety import SafetyMonitor, SafetyStatusLevel

        config = RobotConfig(name="test", n_joints=7)
        monitor = SafetyMonitor(config)

        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            joint_torques=np.zeros(7),
        )

        status = monitor.check_state(state)

        assert status.is_safe
        assert status.level == SafetyStatusLevel.OK
        assert len(status.violations) == 0

    def test_check_velocity_violation(self) -> None:
        """Test detecting velocity violation."""
        from src.deployment.realtime import RobotConfig, RobotState
        from src.deployment.safety import SafetyLimits, SafetyMonitor, SafetyStatusLevel

        config = RobotConfig(name="test", n_joints=7)
        limits = SafetyLimits(
            max_joint_velocity=np.ones(7) * 1.0,
            max_joint_torque=np.ones(7) * 50.0,
        )
        monitor = SafetyMonitor(config, limits)

        # State with excessive velocity
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(7),
            joint_velocities=np.array([0, 0, 0, 2.0, 0, 0, 0]),  # Exceeds limit
            joint_torques=np.zeros(7),
        )

        status = monitor.check_state(state)

        assert not status.is_safe
        assert status.level == SafetyStatusLevel.ERROR
        assert len(status.violations) > 0

    def test_emergency_stop(self) -> None:
        """Test emergency stop functionality."""
        from src.deployment.realtime import RobotConfig
        from src.deployment.safety import SafetyMonitor

        config = RobotConfig(name="test", n_joints=7)
        monitor = SafetyMonitor(config)

        assert not monitor.is_emergency_stopped()

        monitor.trigger_emergency_stop()
        assert monitor.is_emergency_stopped()

        monitor.clear_emergency_stop()
        assert not monitor.is_emergency_stopped()

    def test_speed_override(self) -> None:
        """Test speed override."""
        from src.deployment.realtime import RobotConfig
        from src.deployment.safety import SafetyMonitor

        config = RobotConfig(name="test", n_joints=7)
        monitor = SafetyMonitor(config)

        monitor.set_speed_override(0.5)
        monitor.set_human_nearby(True)

        # Speed should be limited
        assert monitor._speed_override <= 0.5


class TestNearLimitWarnings:
    """Issue #7740: approaching-limit WARNING must be symmetric on both bounds."""

    @staticmethod
    def _monitor_with_limits():
        from src.deployment.realtime import RobotConfig
        from src.deployment.safety import SafetyLimits, SafetyMonitor

        config = RobotConfig(name="test", n_joints=3)
        limits = SafetyLimits(
            max_joint_velocity=np.ones(3) * 10.0,
            max_joint_torque=np.ones(3) * 100.0,
            joint_limits_lower=np.full(3, -1.0),
            joint_limits_upper=np.full(3, 1.0),
        )
        return SafetyMonitor(config, limits)

    @staticmethod
    def _state(positions: np.ndarray):
        from src.deployment.realtime import RobotState

        return RobotState(
            timestamp=0.0,
            joint_positions=positions,
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
        )

    @pytest.mark.unit
    def test_near_lower_limit_warns(self) -> None:
        """A joint just inside the lower limit raises an approaching-limit warning."""
        from src.deployment.safety import SafetyStatusLevel

        monitor = self._monitor_with_limits()
        # -0.95 is within the 0.1 rad margin of the -1.0 lower limit.
        status = monitor.check_state(self._state(np.array([-0.95, 0.0, 0.0])))

        assert status.is_safe
        assert status.level == SafetyStatusLevel.WARNING
        assert any("lower limit" in w for w in status.warnings)

    @pytest.mark.unit
    def test_near_upper_limit_still_warns(self) -> None:
        """The upper-limit warning remains intact alongside the new lower mirror."""
        from src.deployment.safety import SafetyStatusLevel

        monitor = self._monitor_with_limits()
        status = monitor.check_state(self._state(np.array([0.95, 0.0, 0.0])))

        assert status.level == SafetyStatusLevel.WARNING
        assert any("upper limit" in w for w in status.warnings)

    @pytest.mark.unit
    def test_mid_range_no_warning(self) -> None:
        """A joint comfortably inside both limits raises no approaching warnings."""
        from src.deployment.safety import SafetyStatusLevel

        monitor = self._monitor_with_limits()
        status = monitor.check_state(self._state(np.zeros(3)))

        assert status.level == SafetyStatusLevel.OK
        assert status.warnings == []


class TestStoppingDistanceSignature:
    """Issue #7740: get_stopping_distance dropped its unused ``body`` parameter."""

    @staticmethod
    def _monitor():
        from src.deployment.realtime import RobotConfig
        from src.deployment.safety import SafetyMonitor

        return SafetyMonitor(RobotConfig(name="test", n_joints=3))

    @pytest.mark.unit
    def test_stopping_distance_no_body_param(self) -> None:
        """The method is callable with state only and returns a finite angle."""
        from src.deployment.realtime import RobotState

        monitor = self._monitor()
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(3),
            joint_velocities=np.array([2.0, 0.0, 0.0]),
            joint_torques=np.zeros(3),
        )
        # s = omega^2 / (2 * alpha) = 4 / 4 = 1.0 (rad) with alpha=2.0 rad/s^2.
        assert monitor.get_stopping_distance(state) == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.unit
    def test_stopping_distance_rejects_extra_positional_arg(self) -> None:
        """Passing a body name (old signature) now raises a TypeError."""
        from src.deployment.realtime import RobotState

        monitor = self._monitor()
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
        )
        with pytest.raises(TypeError):
            monitor.get_stopping_distance(state, "ee")  # type: ignore[call-arg]

    @pytest.mark.unit
    def test_stopping_distance_zero_velocity(self) -> None:
        """A stationary robot has zero stopping angle."""
        from src.deployment.realtime import RobotState

        monitor = self._monitor()
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(3),
        )
        assert monitor.get_stopping_distance(state) == pytest.approx(0.0)


class TestCollisionAvoidance:
    """Tests for CollisionAvoidance."""

    def test_collision_avoidance_creation(self) -> None:
        """Test creating collision avoidance."""
        from src.deployment.safety import CollisionAvoidance

        class MockEngine:
            def set_joint_positions(self, q) -> None:
                pass

            def get_link_positions(self) -> dict[str, np.ndarray]:
                return {"link_0": np.array([0.5, 0, 0.5])}

        ca = CollisionAvoidance(MockEngine(), safety_distance=0.1)
        assert ca.safety_distance == 0.1

    def test_add_obstacle(self) -> None:
        """Test adding obstacles."""
        from src.deployment.safety import CollisionAvoidance, Obstacle, ObstacleType

        class MockEngine:
            pass

        ca = CollisionAvoidance(MockEngine())

        obstacle = Obstacle(
            name="box1",
            obstacle_type=ObstacleType.BOX,
            position=np.array([1, 0, 0.5]),
            dimensions=np.array([0.2, 0.2, 0.2]),
        )

        ca.add_obstacle(obstacle)
        assert len(ca._obstacles) == 1

        ca.remove_obstacle("box1")
        assert len(ca._obstacles) == 0

    def test_obstacle_distance(self) -> None:
        """Test obstacle distance calculation."""
        from src.deployment.safety import Obstacle, ObstacleType

        # Sphere obstacle
        sphere = Obstacle(
            name="sphere",
            obstacle_type=ObstacleType.SPHERE,
            position=np.array([0, 0, 0]),
            dimensions=np.array([0.5]),  # Radius
            inflation=0.0,
        )

        # Point outside sphere
        point = np.array([1.0, 0, 0])
        dist = sphere.get_distance(point)
        assert dist == pytest.approx(0.5, rel=1e-3)

        # Point inside sphere
        point_inside = np.array([0.2, 0, 0])
        dist_inside = sphere.get_distance(point_inside)
        assert dist_inside < 0

    def test_human_state(self) -> None:
        """Test human state to obstacle conversion."""
        from src.deployment.safety import HumanState, ObstacleType

        human = HumanState(
            position=np.array([1, 0, 1]),
            velocity=np.array([0.1, 0, 0]),
        )

        obstacle = human.to_obstacle()

        assert obstacle.obstacle_type == ObstacleType.HUMAN
        np.testing.assert_array_equal(obstacle.position, human.position)
        assert obstacle.inflation > 0  # Extra margin for humans


class TestIssue2477EStopPositionMode:
    """Issue #2477: E-stop must neutralize position-controlled motion."""

    def _make_monitor_and_state(self) -> tuple:
        from src.deployment.realtime import RobotConfig, RobotState
        from src.deployment.safety import SafetyMonitor

        n = 7
        config = RobotConfig(name="test", n_joints=n)
        monitor = SafetyMonitor(config)
        state = RobotState(
            timestamp=0.0,
            joint_positions=np.ones(n) * 0.5,
            joint_velocities=np.zeros(n),
            joint_torques=np.zeros(n),
        )
        return monitor, state, n

    def test_estop_zeros_position_targets(self) -> None:
        """After E-stop, position targets must not pass through unchanged."""
        from src.deployment.realtime import ControlCommand, ControlMode

        monitor, state, n = self._make_monitor_and_state()
        monitor.trigger_emergency_stop()

        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.POSITION,
            position_targets=np.ones(n) * 2.0,
        )
        safe = monitor.compute_safe_command(cmd, state)

        assert safe.position_targets is not None, (
            "position_targets should be frozen to current position on E-stop"
        )
        np.testing.assert_array_almost_equal(
            safe.position_targets,
            state.joint_positions,
            err_msg=(
                "E-stop must freeze position targets to current joint positions, "
                "not pass desired targets through"
            ),
        )

    def test_estop_zeros_feedforward_torque(self) -> None:
        """After E-stop, feedforward torque must be zeroed."""
        from src.deployment.realtime import ControlCommand, ControlMode

        monitor, state, n = self._make_monitor_and_state()
        monitor.trigger_emergency_stop()

        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.TORQUE,
            feedforward_torque=np.ones(n) * 10.0,
        )
        safe = monitor.compute_safe_command(cmd, state)

        if safe.feedforward_torque is not None:
            np.testing.assert_array_equal(
                safe.feedforward_torque,
                np.zeros(n),
                err_msg="E-stop must zero feedforward torque",
            )

    def test_non_estop_preserves_position_targets(self) -> None:
        """Without E-stop, position targets must pass through (clipped to limits)."""
        from src.deployment.realtime import ControlCommand, ControlMode

        monitor, state, n = self._make_monitor_and_state()
        targets = np.ones(n) * 0.3

        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.POSITION,
            position_targets=targets.copy(),
        )
        safe = monitor.compute_safe_command(cmd, state)

        assert safe.position_targets is not None
        np.testing.assert_array_almost_equal(safe.position_targets, targets)


class TestSafetyMonitorNonFiniteAndShapes:
    """Issue #12972: SafetyMonitor must reject NaN/inf inputs and validate array shapes."""

    @staticmethod
    def _make_monitor(n_joints: int = 3):
        from src.deployment.realtime import RobotConfig
        from src.deployment.safety import SafetyLimits, SafetyMonitor

        config = RobotConfig(
            name="test",
            n_joints=n_joints,
            velocity_limits=np.ones(n_joints) * 2.0,
            torque_limits=np.ones(n_joints) * 50.0,
        )
        limits = SafetyLimits.from_config(config)
        limits.joint_limits_lower = np.ones(n_joints) * -1.0
        limits.joint_limits_upper = np.ones(n_joints) * 1.0
        return SafetyMonitor(config, limits)

    @staticmethod
    def _make_state(
        n_joints: int = 3,
        pos: np.ndarray | None = None,
        vel: np.ndarray | None = None,
        tau: np.ndarray | None = None,
    ):
        from src.deployment.realtime import RobotState

        return RobotState(
            timestamp=0.0,
            joint_positions=pos if pos is not None else np.zeros(n_joints),
            joint_velocities=vel if vel is not None else np.zeros(n_joints),
            joint_torques=tau if tau is not None else np.zeros(n_joints),
        )

    @pytest.mark.parametrize(
        "field_name", ["joint_positions", "joint_velocities", "joint_torques"]
    )
    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_check_state_rejects_nan_and_inf(
        self, field_name: str, bad_val: float
    ) -> None:
        from src.deployment.safety import SafetyStatusLevel

        monitor = self._make_monitor(3)
        kwargs = {
            "joint_positions": np.zeros(3),
            "joint_velocities": np.zeros(3),
            "joint_torques": np.zeros(3),
        }
        arr = np.zeros(3)
        arr[1] = bad_val
        kwargs[field_name] = arr

        from src.deployment.realtime import RobotState

        state = RobotState(timestamp=0.0, **kwargs)
        status = monitor.check_state(state)

        assert status.level == SafetyStatusLevel.ERROR
        assert not status.is_safe
        assert any(field_name in v for v in status.violations)

    @pytest.mark.parametrize("wrong_len", [2, 4])
    @pytest.mark.parametrize(
        "field_name", ["joint_positions", "joint_velocities", "joint_torques"]
    )
    def test_check_state_rejects_wrong_shape_without_crash(
        self, field_name: str, wrong_len: int
    ) -> None:
        from src.deployment.safety import SafetyStatusLevel
        from src.deployment.realtime import RobotState

        monitor = self._make_monitor(3)
        kwargs = {
            "joint_positions": np.zeros(3),
            "joint_velocities": np.zeros(3),
            "joint_torques": np.zeros(3),
        }
        kwargs[field_name] = np.zeros(wrong_len)
        state = RobotState(timestamp=0.0, **kwargs)

        status = monitor.check_state(state)
        assert status.level == SafetyStatusLevel.ERROR
        assert not status.is_safe
        assert any(field_name in v for v in status.violations)

    @pytest.mark.parametrize(
        "field_name",
        [
            "position_targets",
            "velocity_targets",
            "torque_commands",
            "feedforward_torque",
        ],
    )
    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_check_command_rejects_nan_and_inf(
        self, field_name: str, bad_val: float
    ) -> None:
        from src.deployment.realtime import ControlCommand, ControlMode
        from src.deployment.safety import SafetyStatusLevel

        monitor = self._make_monitor(3)
        kwargs = {
            "position_targets": np.zeros(3),
            "velocity_targets": np.zeros(3),
            "torque_commands": np.zeros(3),
            "feedforward_torque": np.zeros(3),
        }
        arr = np.zeros(3)
        arr[0] = bad_val
        kwargs[field_name] = arr

        cmd = ControlCommand(timestamp=0.0, mode=ControlMode.POSITION, **kwargs)
        status = monitor.check_command(cmd)

        assert status.level == SafetyStatusLevel.ERROR
        assert not status.is_safe
        assert any(field_name in v for v in status.violations)

    @pytest.mark.parametrize("wrong_len", [1, 5])
    @pytest.mark.parametrize(
        "field_name",
        [
            "position_targets",
            "velocity_targets",
            "torque_commands",
            "feedforward_torque",
        ],
    )
    def test_check_command_rejects_wrong_shape_without_crash(
        self, field_name: str, wrong_len: int
    ) -> None:
        from src.deployment.realtime import ControlCommand, ControlMode
        from src.deployment.safety import SafetyStatusLevel

        monitor = self._make_monitor(3)
        kwargs = {
            "position_targets": np.zeros(3),
            "velocity_targets": np.zeros(3),
            "torque_commands": np.zeros(3),
            "feedforward_torque": np.zeros(3),
        }
        kwargs[field_name] = np.zeros(wrong_len)

        cmd = ControlCommand(timestamp=0.0, mode=ControlMode.POSITION, **kwargs)
        status = monitor.check_command(cmd)

        assert status.level == SafetyStatusLevel.ERROR
        assert not status.is_safe
        assert any(field_name in v for v in status.violations)

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_compute_safe_command_sanitizes_nan_and_inf(self, bad_val: float) -> None:
        from src.deployment.realtime import ControlCommand, ControlMode

        monitor = self._make_monitor(3)
        state = self._make_state(3, pos=np.array([0.2, -0.3, 0.5]))

        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.POSITION,
            position_targets=np.array([bad_val, 0.0, 0.0]),
            velocity_targets=np.array([bad_val, 1.0, 0.0]),
            torque_commands=np.array([bad_val, 0.0, -10.0]),
            feedforward_torque=np.array([bad_val, 0.0, 0.0]),
        )

        safe = monitor.compute_safe_command(cmd, state)

        # Output arrays must all be strictly finite
        assert safe.position_targets is not None
        assert safe.velocity_targets is not None
        assert safe.torque_commands is not None
        assert np.all(np.isfinite(safe.position_targets))
        assert np.all(np.isfinite(safe.velocity_targets))
        assert np.all(np.isfinite(safe.torque_commands))

        # Finite positions must be held safely within joint limits
        np.testing.assert_allclose(safe.position_targets, state.joint_positions)
        # Velocities and torques must be zeroed for safety on invalid inputs
        assert safe.velocity_targets[0] == 0.0
        assert safe.torque_commands[0] == 0.0

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_set_speed_override_rejects_non_finite(self, bad_val: float) -> None:
        monitor = self._make_monitor(3)
        with pytest.raises(ValueError, match="finite"):
            monitor.set_speed_override(bad_val)
