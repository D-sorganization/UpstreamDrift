"""Tests for Left-Handed Player Support.

Guideline B6 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from shared.python.handedness_support import (
    SAGITTAL_MIRROR,
    Handedness,
    HandednessConverter,
    detect_handedness_from_metadata,
    mirror_angular_velocity,
    mirror_joint_configuration,
    mirror_position,
    mirror_rotation_matrix,
    mirror_trajectory,
    mirror_velocity,
    validate_energy_conservation,
    validate_mirror_trajectory,
)


class TestMirrorPosition:
    """Tests for position mirroring."""

    def test_y_coordinate_flips(self) -> None:
        """Y coordinate should flip under sagittal mirror."""
        pos = np.array([1.0, 2.0, 3.0])
        mirrored = mirror_position(pos)

        assert mirrored[0] == pytest.approx(1.0)
        assert mirrored[1] == pytest.approx(-2.0)
        assert mirrored[2] == pytest.approx(3.0)

    def test_origin_unchanged(self) -> None:
        """Origin should be unchanged under mirroring."""
        origin = np.array([0.0, 0.0, 0.0])
        mirrored = mirror_position(origin)

        np.testing.assert_allclose(mirrored, origin)

    def test_trajectory_mirroring(self) -> None:
        """Trajectory (N, 3) should mirror correctly."""
        trajectory = np.array(
            [
                [1.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [3.0, 3.0, 0.0],
            ]
        )

        mirrored = mirror_position(trajectory)

        assert mirrored.shape == trajectory.shape
        np.testing.assert_allclose(mirrored[:, 0], trajectory[:, 0])
        np.testing.assert_allclose(mirrored[:, 1], -trajectory[:, 1])
        np.testing.assert_allclose(mirrored[:, 2], trajectory[:, 2])


class TestMirrorVelocity:
    """Tests for velocity mirroring."""

    def test_y_velocity_flips(self) -> None:
        """Y velocity component should flip."""
        vel = np.array([10.0, 5.0, 3.0])
        mirrored = mirror_velocity(vel)

        assert mirrored[0] == pytest.approx(10.0)
        assert mirrored[1] == pytest.approx(-5.0)
        assert mirrored[2] == pytest.approx(3.0)


class TestMirrorRotation:
    """Tests for rotation matrix mirroring."""

    def test_identity_rotates_to_reflected_identity(self) -> None:
        """Identity rotation should produce flipped identity."""
        R = np.eye(3)
        R_mirrored = mirror_rotation_matrix(R)

        # M @ I @ M = M @ M = I (since M is involutory for our mirror)
        expected = SAGITTAL_MIRROR @ np.eye(3) @ SAGITTAL_MIRROR
        np.testing.assert_allclose(R_mirrored, expected, atol=1e-10)

    def test_mirrored_rotation_is_orthogonal(self) -> None:
        """Mirrored rotation should remain orthogonal."""
        # Create a rotation about Z axis
        angle = np.pi / 4
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

        R_mirrored = mirror_rotation_matrix(R)

        # Check orthogonality
        np.testing.assert_allclose(R_mirrored @ R_mirrored.T, np.eye(3), atol=1e-10)


class TestMirrorAngularVelocity:
    """Tests for angular velocity mirroring."""

    def test_pseudovector_behavior(self) -> None:
        """Angular velocity should behave as pseudovector under reflection."""
        omega = np.array([1.0, 2.0, 3.0])
        mirrored = mirror_angular_velocity(omega)

        # X and Z flip, Y preserves
        assert mirrored[0] == pytest.approx(-1.0)
        assert mirrored[1] == pytest.approx(2.0)
        assert mirrored[2] == pytest.approx(-3.0)


class TestMirrorJointConfiguration:
    """Tests for joint configuration mirroring."""

    def test_y_axis_revolute_preserves(self) -> None:
        """Revolute about Y should preserve sign."""
        q = np.array([0.5])
        joint_types = ["revolute"]
        joint_axes = [np.array([0.0, 1.0, 0.0])]

        q_mirrored = mirror_joint_configuration(q, joint_types, joint_axes)

        assert q_mirrored[0] == pytest.approx(0.5)

    def test_z_axis_revolute_flips(self) -> None:
        """Revolute about Z should flip sign."""
        q = np.array([0.5])
        joint_types = ["revolute"]
        joint_axes = [np.array([0.0, 0.0, 1.0])]

        q_mirrored = mirror_joint_configuration(q, joint_types, joint_axes)

        assert q_mirrored[0] == pytest.approx(-0.5)


class TestMirrorTrajectory:
    """Tests for complete trajectory mirroring."""

    def test_complete_trajectory_mirror(self) -> None:
        """Complete trajectory should mirror all components."""
        positions = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
        velocities = np.array([[10.0, 5.0, 0.0], [10.0, 5.0, 0.0]])
        orientations = np.array([np.eye(3), np.eye(3)])
        angular_velocities = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

        result = mirror_trajectory(
            positions, velocities, orientations, angular_velocities
        )

        assert "positions" in result
        assert "velocities" in result
        assert "orientations" in result
        assert "angular_velocities" in result

        # Check Y flip for positions
        np.testing.assert_allclose(result["positions"][:, 1], -positions[:, 1])


class TestHandednessDetection:
    """Tests for handedness detection from metadata."""

    def test_detects_left_handed(self) -> None:
        """Should detect left-handed from metadata."""
        metadata = {"handedness": "left"}
        result = detect_handedness_from_metadata(metadata)
        assert result == Handedness.LEFT_HANDED

    def test_detects_right_handed(self) -> None:
        """Should detect right-handed from metadata."""
        metadata = {"handedness": "right"}
        result = detect_handedness_from_metadata(metadata)
        assert result == Handedness.RIGHT_HANDED

    def test_defaults_to_right(self) -> None:
        """Should default to right-handed when no metadata."""
        metadata: dict[str, object] = {}
        result = detect_handedness_from_metadata(metadata)
        assert result == Handedness.RIGHT_HANDED

    def test_detects_from_boolean_flag(self) -> None:
        """Should detect from is_left_handed flag."""
        metadata = {"is_left_handed": True}
        result = detect_handedness_from_metadata(metadata)
        assert result == Handedness.LEFT_HANDED


class TestValidation:
    """Tests for trajectory and energy validation."""

    def test_valid_mirror_trajectory(self) -> None:
        """Properly mirrored trajectory should validate."""
        original = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]])
        mirrored = mirror_position(original)

        result = validate_mirror_trajectory(original, mirrored)

        assert result["valid"]
        assert result["y_flipped"]
        assert result["x_preserved"]
        assert result["z_preserved"]
        assert result["path_length_preserved"]

    def test_energy_conservation(self) -> None:
        """Kinetic energy should be preserved under mirroring."""
        original_v = np.array([[10.0, 5.0, 3.0], [10.0, 5.0, 3.0]])
        mirrored_v = mirror_velocity(original_v)

        result = validate_energy_conservation(original_v, mirrored_v)

        assert result["valid"]
        assert result["original_total_ke"] == pytest.approx(result["mirrored_total_ke"])


class TestHandednessConverter:
    """Tests for HandednessConverter class."""

    def test_no_conversion_same_handedness(self) -> None:
        """No conversion when source and target match."""
        converter = HandednessConverter(Handedness.RIGHT_HANDED)
        positions = np.array([[1.0, 2.0, 3.0]])

        result = converter.convert_to(Handedness.RIGHT_HANDED, positions)

        np.testing.assert_allclose(result["positions"], positions)

    def test_conversion_different_handedness(self) -> None:
        """Should convert when handedness differs."""
        converter = HandednessConverter(Handedness.RIGHT_HANDED)
        positions = np.array([[1.0, 2.0, 3.0]])

        result = converter.convert_to(Handedness.LEFT_HANDED, positions)

        # Y should be flipped
        assert result["positions"][0, 1] == pytest.approx(-2.0)

    def test_is_conversion_needed(self) -> None:
        """Should correctly detect when conversion is needed."""
        converter = HandednessConverter(Handedness.RIGHT_HANDED)

        assert not converter.is_conversion_needed(Handedness.RIGHT_HANDED)
        assert converter.is_conversion_needed(Handedness.LEFT_HANDED)
