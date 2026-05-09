"""
Unit tests for 6DOF positioning module.

Tests for Pose6DOF, Transform6DOF, and EntityPlacement classes
following TDD principles - tests written first.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.spatial_algebra.pose6dof import (
    EntityPlacement,
    PlacementGroup,
    Pose6DOF,
    Transform6DOF,
    axis_angle_to_rotation_matrix,
    euler_to_quaternion,
    euler_to_rotation_matrix,
    quaternion_multiply,
    quaternion_to_euler,
    rotation_matrix_to_euler,
)


class TestPose6DOF:
    """Tests for Pose6DOF class - intuitive 6DOF positioning."""

    def test_create_identity_pose(self) -> None:
        """Test creating an identity pose at origin with no rotation."""
        pose = Pose6DOF()
        np.testing.assert_allclose(pose.position, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(pose.euler_angles, [0, 0, 0], atol=1e-10)

    def test_create_pose_with_position(self) -> None:
        """Test creating a pose with specified position."""
        pose = Pose6DOF(position=[1.0, 2.0, 3.0])
        np.testing.assert_allclose(pose.position, [1, 2, 3], atol=1e-10)
        np.testing.assert_allclose(pose.euler_angles, [0, 0, 0], atol=1e-10)

    def test_create_pose_with_euler_angles(self) -> None:
        """Test creating a pose with roll, pitch, yaw."""
        roll, pitch, yaw = np.pi / 6, np.pi / 4, np.pi / 3
        pose = Pose6DOF(euler_angles=[roll, pitch, yaw])
        np.testing.assert_allclose(pose.position, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(pose.euler_angles, [roll, pitch, yaw], atol=1e-10)

    def test_create_pose_with_quaternion(self) -> None:
        """Test creating a pose from quaternion."""
        # Quaternion for 90° rotation about z-axis: [w, x, y, z]
        quat = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
        pose = Pose6DOF.from_quaternion([0, 0, 0], quat)
        # Should have yaw of 90°
        np.testing.assert_allclose(pose.euler_angles[2], np.pi / 2, atol=1e-6)

    def test_pose_to_quaternion_roundtrip(self) -> None:
        """Test euler -> quaternion -> euler roundtrip."""
        original = Pose6DOF(euler_angles=[0.1, 0.2, 0.3])
        quat = original.to_quaternion()
        reconstructed = Pose6DOF.from_quaternion([0, 0, 0], quat)
        np.testing.assert_allclose(
            original.euler_angles, reconstructed.euler_angles, atol=1e-10
        )

    def test_pose_rotation_matrix(self) -> None:
        """Test conversion to 3x3 rotation matrix."""
        # 90° rotation about z-axis
        pose = Pose6DOF(euler_angles=[0, 0, np.pi / 2])
        R = pose.rotation_matrix
        assert R.shape == (3, 3)

        # Check rotation is orthogonal
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

        # Check specific rotation effect
        x_axis = np.array([1, 0, 0])
        rotated = R @ x_axis
        np.testing.assert_allclose(rotated, [0, 1, 0], atol=1e-10)

    def test_pose_homogeneous_matrix(self) -> None:
        """Test conversion to 4x4 homogeneous transform."""
        pose = Pose6DOF(position=[1, 2, 3], euler_angles=[0, 0, np.pi / 2])
        T = pose.homogeneous_matrix
        assert T.shape == (4, 4)

        # Check structure
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)
        np.testing.assert_allclose(T[:3, 3], [1, 2, 3], atol=1e-10)

    def test_pose_translate(self) -> None:
        """Test translation operation."""
        pose = Pose6DOF(position=[1, 2, 3])
        translated = pose.translate([1, 0, 0])
        np.testing.assert_allclose(translated.position, [2, 2, 3], atol=1e-10)
        # Original unchanged
        np.testing.assert_allclose(pose.position, [1, 2, 3], atol=1e-10)

    def test_pose_rotate_euler(self) -> None:
        """Test rotation by euler angles."""
        pose = Pose6DOF()
        rotated = pose.rotate_euler([0, 0, np.pi / 2])
        np.testing.assert_allclose(rotated.euler_angles[2], np.pi / 2, atol=1e-10)

    def test_pose_x_y_z_properties(self) -> None:
        """Test convenient x, y, z accessors."""
        pose = Pose6DOF(position=[1.5, 2.5, 3.5])
        assert pose.x == pytest.approx(1.5)
        assert pose.y == pytest.approx(2.5)
        assert pose.z == pytest.approx(3.5)

    def test_pose_roll_pitch_yaw_properties(self) -> None:
        """Test convenient roll, pitch, yaw accessors."""
        roll, pitch, yaw = 0.1, 0.2, 0.3
        pose = Pose6DOF(euler_angles=[roll, pitch, yaw])
        assert pose.roll == pytest.approx(roll)
        assert pose.pitch == pytest.approx(pitch)
        assert pose.yaw == pytest.approx(yaw)

    def test_pose_set_position_components(self) -> None:
        """Test setting individual position components."""
        pose = Pose6DOF()
        pose.x = 5.0
        pose.y = 6.0
        pose.z = 7.0
        np.testing.assert_allclose(pose.position, [5, 6, 7], atol=1e-10)

    def test_pose_set_rotation_components(self) -> None:
        """Test setting individual rotation components."""
        pose = Pose6DOF()
        pose.roll = 0.1
        pose.pitch = 0.2
        pose.yaw = 0.3
        np.testing.assert_allclose(pose.euler_angles, [0.1, 0.2, 0.3], atol=1e-10)

    def test_pose_inverse(self) -> None:
        """Test pose inversion."""
        pose = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        inv = pose.inverse()

        # Composing with inverse should give identity
        composed = pose.compose(inv)
        np.testing.assert_allclose(composed.position, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(composed.euler_angles, [0, 0, 0], atol=1e-6)

    def test_pose_compose(self) -> None:
        """Test composing two poses."""
        pose1 = Pose6DOF(position=[1, 0, 0])
        pose2 = Pose6DOF(position=[0, 1, 0])
        composed = pose1.compose(pose2)
        np.testing.assert_allclose(composed.position, [1, 1, 0], atol=1e-10)

    def test_pose_compose_with_rotation(self) -> None:
        """Test composing poses with rotation."""
        # First rotate 90° about z, then translate [1, 0, 0] in local frame
        pose1 = Pose6DOF(euler_angles=[0, 0, np.pi / 2])
        pose2 = Pose6DOF(position=[1, 0, 0])
        composed = pose1.compose(pose2)

        # After 90° z rotation, local x becomes world y
        np.testing.assert_allclose(composed.position, [0, 1, 0], atol=1e-10)

    def test_pose_transform_point(self) -> None:
        """Test transforming a point by the pose."""
        pose = Pose6DOF(position=[1, 0, 0], euler_angles=[0, 0, np.pi / 2])
        point = np.array([1, 0, 0])
        transformed = pose.transform_point(point)
        # Rotate [1,0,0] by 90° about z -> [0,1,0], then translate by [1,0,0]
        np.testing.assert_allclose(transformed, [1, 1, 0], atol=1e-10)

    def test_pose_transform_vector(self) -> None:
        """Test transforming a direction vector (no translation)."""
        pose = Pose6DOF(position=[10, 20, 30], euler_angles=[0, 0, np.pi / 2])
        vector = np.array([1, 0, 0])
        transformed = pose.transform_vector(vector)
        # Only rotation, no translation
        np.testing.assert_allclose(transformed, [0, 1, 0], atol=1e-10)

    def test_pose_equality(self) -> None:
        """Test pose equality comparison."""
        pose1 = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        pose2 = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        pose3 = Pose6DOF(position=[1, 2, 4], euler_angles=[0.1, 0.2, 0.3])

        assert pose1 == pose2
        assert pose1 != pose3

    def test_pose_copy(self) -> None:
        """Test pose copying."""
        original = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        copied = original.copy()

        assert original == copied
        # Modifying copy shouldn't affect original
        copied.x = 999
        assert original.x == pytest.approx(1.0)

    def test_pose_to_spatial_transform(self) -> None:
        """Test conversion to 6x6 Plücker transform matrix."""
        pose = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        X = pose.to_spatial_transform()
        assert X.shape == (6, 6)

        # Verify structure: should be consistent with xtrans
        R = pose.rotation_matrix
        # Upper left 3x3 should be rotation
        np.testing.assert_allclose(X[:3, :3], R, atol=1e-10)
        # Lower right 3x3 should be rotation
        np.testing.assert_allclose(X[3:6, 3:6], R, atol=1e-10)

    def test_pose_repr(self) -> None:
        """Test string representation."""
        pose = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        repr_str = repr(pose)
        assert "Pose6DOF" in repr_str
        assert "position" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
