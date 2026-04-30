"""Tests for PendulumScrewKinematics (Guideline C3 - Required)."""

import math

import numpy as np
import pytest

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
)
from src.engines.physics_engines.pendulum.python.pendulum_screw_kinematics import (
    PendulumScrewKinematics,
)
from src.shared.python.screw_theory import ScrewAxis, Twist


@pytest.fixture
def dynamics() -> DoublePendulumDynamics:
    """Default double pendulum dynamics instance."""
    return DoublePendulumDynamics()


@pytest.fixture
def sk(dynamics: DoublePendulumDynamics) -> PendulumScrewKinematics:
    """PendulumScrewKinematics built from default dynamics."""
    return PendulumScrewKinematics(dynamics)


class TestPositionComputation:
    def test_arm_tip_at_rest_points_down(self, sk: PendulumScrewKinematics) -> None:
        """At θ1=0 the arm tip should be directly below the shoulder."""
        p = sk.compute_arm_tip_position(0.0)
        assert abs(p[0]) < 1e-10, "x should be zero at rest"
        assert p[1] < 0, "y should be negative (below pivot)"
        assert abs(p[2]) < 1e-10, "z should be zero (planar)"

    def test_arm_tip_length_is_l1(self, sk: PendulumScrewKinematics) -> None:
        """Distance from origin to arm tip must equal L1 for any angle."""
        for theta in [0.0, math.pi / 6, math.pi / 3, math.pi / 2]:
            p = sk.compute_arm_tip_position(theta)
            assert abs(np.linalg.norm(p) - sk._l1) < 1e-9

    def test_clubhead_at_rest_points_straight_down(
        self, sk: PendulumScrewKinematics
    ) -> None:
        """At θ1=θ2=0 the clubhead is L1+L2 below the shoulder."""
        p = sk.compute_clubhead_position(0.0, 0.0)
        assert abs(p[0]) < 1e-10
        assert abs(p[1] + sk._l1 + sk._l2) < 1e-9


class TestTwistComputation:
    def test_twist_at_rest_is_zero(self, sk: PendulumScrewKinematics) -> None:
        q = np.array([0.0, 0.0])
        v = np.array([0.0, 0.0])

        for body in (sk.BODY_ARM_TIP, sk.BODY_CLUBHEAD):
            twist = sk.compute_twist(q, v, body)
            assert np.allclose(twist.angular, 0, atol=1e-12)
            assert np.allclose(twist.linear, 0, atol=1e-12)

    def test_twist_returns_twist_instance(self, sk: PendulumScrewKinematics) -> None:
        q = np.array([0.1, 0.2])
        v = np.array([1.0, 0.5])
        twist = sk.compute_twist(q, v, sk.BODY_ARM_TIP)
        assert isinstance(twist, Twist)

    def test_arm_tip_angular_velocity_matches_omega1(
        self, sk: PendulumScrewKinematics
    ) -> None:
        """Arm tip angular velocity must equal ω1 (about z)."""
        q = np.array([0.3, 0.0])
        v = np.array([2.5, 0.0])
        twist = sk.compute_twist(q, v, sk.BODY_ARM_TIP)
        assert abs(twist.angular[2] - 2.5) < 1e-10
        assert abs(twist.angular[0]) < 1e-10
        assert abs(twist.angular[1]) < 1e-10

    def test_clubhead_angular_velocity_matches_omega1_plus_omega2(
        self, sk: PendulumScrewKinematics
    ) -> None:
        """Clubhead angular velocity must equal ω1+ω2."""
        q = np.array([0.3, 0.5])
        v = np.array([1.0, 2.0])
        twist = sk.compute_twist(q, v, sk.BODY_CLUBHEAD)
        assert abs(twist.angular[2] - 3.0) < 1e-10

    def test_linear_velocity_perpendicular_to_radius(
        self, sk: PendulumScrewKinematics
    ) -> None:
        """For pure rotation the linear velocity must be ⊥ to the position vector."""
        q = np.array([math.pi / 4, 0.0])
        v = np.array([1.0, 0.0])
        twist = sk.compute_twist(q, v, sk.BODY_ARM_TIP)
        # dot(v, r) == 0 for rigid rotation
        dot = np.dot(twist.linear, twist.reference_point)
        assert abs(dot) < 1e-9

    def test_unknown_body_raises_value_error(self, sk: PendulumScrewKinematics) -> None:
        with pytest.raises(ValueError, match="Unknown body"):
            sk.compute_twist(np.zeros(2), np.zeros(2), "nonexistent")


class TestScrewAxis:
    def test_screw_axis_returns_screw_axis_instance(
        self, sk: PendulumScrewKinematics
    ) -> None:
        q = np.array([0.2, 0.3])
        v = np.array([1.0, 0.5])
        twist = sk.compute_twist(q, v, sk.BODY_ARM_TIP)
        screw = sk.compute_screw_axis(twist)
        assert isinstance(screw, ScrewAxis)

    def test_rotating_pendulum_screw_axis_direction_is_z(
        self, sk: PendulumScrewKinematics
    ) -> None:
        """For planar rotation the ISA must be along ±ẑ."""
        q = np.array([0.4, 0.0])
        v = np.array([3.0, 0.0])
        twist = sk.compute_twist(q, v, sk.BODY_ARM_TIP)
        screw = sk.compute_screw_axis(twist)
        assert not screw.is_singular
        assert abs(abs(screw.axis_direction[2]) - 1.0) < 1e-9
        assert abs(screw.axis_direction[0]) < 1e-9
        assert abs(screw.axis_direction[1]) < 1e-9

    def test_zero_velocity_is_singular(self, sk: PendulumScrewKinematics) -> None:
        q = np.zeros(2)
        v = np.zeros(2)
        twist = sk.compute_twist(q, v, sk.BODY_CLUBHEAD)
        screw = sk.compute_screw_axis(twist)
        assert screw.is_singular

    def test_pure_rotation_pitch_is_near_zero(
        self, sk: PendulumScrewKinematics
    ) -> None:
        """Planar rotation has zero pitch (no translation along axis)."""
        q = np.array([math.pi / 3, 0.0])
        v = np.array([2.0, 0.0])
        twist = sk.compute_twist(q, v, sk.BODY_ARM_TIP)
        screw = sk.compute_screw_axis(twist)
        assert not screw.is_singular
        # dot(ω, v) = 0 for planar rotation → pitch = 0
        assert abs(screw.pitch) < 1e-9


class TestAnalyzeKeyPoints:
    def test_returns_both_bodies(self, sk: PendulumScrewKinematics) -> None:
        q = np.array([0.2, 0.4])
        v = np.array([1.0, -0.5])
        results = sk.analyze_key_points(q, v)
        assert sk.BODY_ARM_TIP in results
        assert sk.BODY_CLUBHEAD in results

    def test_each_entry_is_twist_screw_pair(self, sk: PendulumScrewKinematics) -> None:
        q = np.array([0.1, 0.2])
        v = np.array([0.5, 0.3])
        results = sk.analyze_key_points(q, v)
        for body, (twist, screw) in results.items():
            assert isinstance(twist, Twist), f"Expected Twist for {body}"
            assert isinstance(screw, ScrewAxis), f"Expected ScrewAxis for {body}"


class TestVisualizeScrewAxis:
    def test_returns_two_points(self, sk: PendulumScrewKinematics) -> None:
        q = np.array([0.3, 0.1])
        v = np.array([1.5, 0.5])
        twist = sk.compute_twist(q, v, sk.BODY_CLUBHEAD)
        screw = sk.compute_screw_axis(twist)
        start, end = sk.visualize_screw_axis(screw, length=0.5)
        assert start.shape == (3,)
        assert end.shape == (3,)
        assert abs(np.linalg.norm(end - start) - 0.5) < 1e-9
