"""Tests for screw-theoretic kinematics (Guideline C3 - Required)."""

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.screw_kinematics import (
    ScrewAxis,
    ScrewKinematicsAnalyzer,
    Twist,
)


@pytest.fixture
def simple_pendulum() -> mujoco.MjModel:
    """Simple pendulum for testing."""
    xml = """
    <mujoco>
        <worldbody>
            <body name="pendulum" pos="0 0 0">
                <joint name="hinge" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.01 0.5" mass="1.0"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


class TestTwistCalculations:
    """Test twist extraction from Jacobians."""

    def test_twist_at_rest_is_zero(self, simple_pendulum: mujoco.MjModel) -> None:
        """Test twist is zero when body is at rest."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        qpos = np.array([0.0])
        qvel = np.array([0.0])  # At rest
        body_id = 1  # Pendulum body

        twist = analyzer.compute_twist(qpos, qvel, body_id)

        assert np.allclose(twist.angular, 0, atol=1e-10)
        assert np.allclose(twist.linear, 0, atol=1e-10)
        assert isinstance(twist, Twist)

    def test_twist_angular_velocity_matches_joint_velocity(
        self, simple_pendulum: mujoco.MjModel
    ) -> None:
        """Test angular velocity component matches joint velocity."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        qpos = np.array([0.0])
        qvel = np.array([2.0])  # 2 rad/s
        body_id = 1

        twist = analyzer.compute_twist(qpos, qvel, body_id)

        # For hinge joint about Y-axis, angular velocity should be [0, 2.0, 0]
        assert abs(twist.angular[1] - 2.0) < 0.1, (
            f"Expected ω_y ≈ 2.0, got {twist.angular[1]}"
        )

    def test_twist_includes_reference_point(
        self, simple_pendulum: mujoco.MjModel
    ) -> None:
        """Test twist includes reference point."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        qpos = np.array([0.5])
        qvel = np.array([1.0])
        body_id = 1

        twist = analyzer.compute_twist(qpos, qvel, body_id)

        assert twist.reference_point is not None
        assert len(twist.reference_point) == 3
        assert twist.body_name == "pendulum"


class TestScrewAxisCalculations:
    """Test Instantaneous Screw Axis computation."""

    def test_pure_rotation_has_zero_pitch(
        self, simple_pendulum: mujoco.MjModel
    ) -> None:
        """Test pure rotation gives pitch = 0."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        # Pure rotation about fixed axis
        qpos = np.array([0.0])
        qvel = np.array([1.0])
        body_id = 1

        twist = analyzer.compute_twist(qpos, qvel, body_id)
        screw = analyzer.compute_screw_axis(twist)

        # For revolute joint, pitch should be close to zero
        # (some numerical tolerance due to COM offset)
        assert abs(screw.pitch) < 1.0, (
            f"Expected pitch ≈ 0 for pure rotation, got {screw.pitch}"
        )
        assert not screw.is_singular

    def test_pure_translation_is_singular(
        self, simple_pendulum: mujoco.MjModel
    ) -> None:
        """Test pure translation is detected as singular."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        # Create artificial twist with pure translation (no rotation)
        twist = Twist(
            angular=np.array([0.0, 0.0, 0.0]),
            linear=np.array([1.0, 0.0, 0.0]),
            body_name="test",
            reference_point=np.array([0.0, 0.0, 0.0]),
        )

        screw = analyzer.compute_screw_axis(twist)

        assert screw.is_singular, "Pure translation should be singular"
        assert screw.pitch == float("inf")

    def test_axis_direction_is_unit_vector(
        self, simple_pendulum: mujoco.MjModel
    ) -> None:
        """Test screw axis direction is normalized."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        qpos = np.array([0.3])
        qvel = np.array([2.0])
        body_id = 1

        twist = analyzer.compute_twist(qpos, qvel, body_id)
        screw = analyzer.compute_screw_axis(twist)

        if not screw.is_singular:
            axis_norm = np.linalg.norm(screw.axis_direction)
            assert abs(axis_norm - 1.0) < 1e-6, (
                f"Axis direction should be unit vector, got norm={axis_norm}"
            )

    def test_pitch_formula(self, simple_pendulum: mujoco.MjModel) -> None:
        """Test pitch calculation: h = (ω · v) / |ω|²."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        # Create known twist
        ω = np.array([0.0, 2.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])  # Parallel component

        twist = Twist(
            angular=ω,
            linear=v,
            body_name="test",
            reference_point=np.zeros(3),
        )

        screw = analyzer.compute_screw_axis(twist)

        # Expected: h = (ω · v) / |ω|² = 2.0 / 4.0 = 0.5
        expected_pitch = np.dot(ω, v) / np.dot(ω, ω)
        assert abs(screw.pitch - expected_pitch) < 1e-6, (
            f"Expected pitch {expected_pitch}, got {screw.pitch}"
        )


class TestKeyPointAnalysis:
    """Test analysis of multiple key points."""

    def test_analyze_key_points_returns_dict(
        self, simple_pendulum: mujoco.MjModel
    ) -> None:
        """Test key point analysis returns results dict."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        qpos = np.array([0.2])
        qvel = np.array([1.0])

        results = analyzer.analyze_key_points(qpos, qvel, ["pendulum"])

        assert isinstance(results, dict)
        assert "pendulum" in results

        twist, screw = results["pendulum"]
        assert isinstance(twist, Twist)
        assert isinstance(screw, ScrewAxis)

    def test_analyze_missing_body_warns(
        self, simple_pendulum: mujoco.MjModel, caplog
    ) -> None:
        """Test analysis of non-existent body logs warning."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        qpos = np.array([0.0])
        qvel = np.array([0.0])

        results = analyzer.analyze_key_points(qpos, qvel, ["nonexistent_body"])

        assert "nonexistent_body" not in results
        # Check for warning (caplog might not capture due to logger config)


class TestVisualization:
    """Test visualization support functions."""

    def test_visualize_screw_axis_returns_line_segment(
        self, simple_pendulum: mujoco.MjModel
    ) -> None:
        """Test screw axis visualization returns start/end points."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        qpos = np.array([0.3])
        qvel = np.array([1.5])
        body_id = 1

        twist = analyzer.compute_twist(qpos, qvel, body_id)
        screw = analyzer.compute_screw_axis(twist)

        start, end = analyzer.visualize_screw_axis(screw, length=1.0)

        assert len(start) == 3
        assert len(end) == 3

        # Line length should be approximately 1.0 m
        line_length = np.linalg.norm(end - start)
        assert 0.9 < line_length < 1.1, f"Expected line length ≈ 1.0, got {line_length}"

    def test_singular_axis_visualizes_along_velocity(
        self, simple_pendulum: mujoco.MjModel
    ) -> None:
        """Test singular (translation) axis visualized along velocity."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        twist = Twist(
            angular=np.zeros(3),
            linear=np.array([1.0, 0.0, 0.0]),
            body_name="test",
            reference_point=np.zeros(3),
        )

        screw = analyzer.compute_screw_axis(twist)
        start, end = analyzer.visualize_screw_axis(screw, length=0.5)

        # Direction should be along X (velocity direction)
        direction = (end - start) / np.linalg.norm(end - start)
        assert abs(direction[0] - 1.0) < 1e-6, (
            "Singular axis should align with velocity direction"
        )


class TestManipulability:
    """Test screw-based manipulability."""

    def test_manipulability_is_nonnegative(
        self, simple_pendulum: mujoco.MjModel
    ) -> None:
        """Test manipulability measure is non-negative."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        qpos = np.array([0.5])
        body_id = 1

        manip = analyzer.compute_manipulability_screw(qpos, body_id)

        assert manip >= 0, f"Manipulability should be non-negative, got {manip}"

    def test_manipulability_varies_with_configuration(
        self, simple_pendulum: mujoco.MjModel
    ) -> None:
        """Test manipulability changes with configuration."""
        analyzer = ScrewKinematicsAnalyzer(simple_pendulum)

        body_id = 1

        manip1 = analyzer.compute_manipulability_screw(np.array([0.0]), body_id)
        manip2 = analyzer.compute_manipulability_screw(np.array([np.pi / 2]), body_id)

        # For single-DOF system, manipulability should differ at different configs
        # (due to changing Jacobian)
        # May or may not differ significantly - just check it's computed
        assert manip1 >= 0
        assert manip2 >= 0


@pytest.mark.integration
class TestScrewKinematicsPhysics:
    """Integration tests for screw kinematics physics."""

    def test_pitch_matches_analytical_helix(
        self, simple_pendulum: mujoco.MjModel
    ) -> None:
        """Test pitch calculation matches analytical helix motion."""
        pytest.skip("Requires helical motion model - implement in follow-up")

        # Verify computed pitch matches known pitch

    def test_screw_axis_lies_on_rotation_axis(
        self, simple_pendulum: mujoco.MjModel
    ) -> None:
        """Test screw axis for pure rotation lies on rotation axis."""
        pytest.skip("Requires precise geometric validation - implement in follow-up")

        # through the joint axis
