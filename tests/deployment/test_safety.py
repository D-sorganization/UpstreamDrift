"""Tests for safety module."""

from __future__ import annotations

import numpy as np
import pytest


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


class TestCollisionAvoidance:
    """Tests for CollisionAvoidance."""

    def test_collision_avoidance_creation(self) -> None:
        """Test creating collision avoidance."""
        from src.deployment.safety import CollisionAvoidance

        class MockEngine:
            def set_joint_positions(self, q):
                pass

            def get_link_positions(self):
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
