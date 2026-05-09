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


class TestTransform6DOF:
    """Tests for Transform6DOF class - 6DOF rigid body transformations."""

    def test_pose6dof_placement_identity_transform(self) -> None:
        """Test identity transformation."""
        T = Transform6DOF.identity()
        np.testing.assert_allclose(T.translation, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(T.rotation_matrix, np.eye(3), atol=1e-10)

    def test_translation_only(self) -> None:
        """Test pure translation transform."""
        T = Transform6DOF.from_translation([1, 2, 3])
        np.testing.assert_allclose(T.translation, [1, 2, 3], atol=1e-10)
        np.testing.assert_allclose(T.rotation_matrix, np.eye(3), atol=1e-10)

    def test_rotation_about_x(self) -> None:
        """Test rotation about x-axis."""
        T = Transform6DOF.from_rotation_x(np.pi / 2)
        R = T.rotation_matrix

        # y -> z for 90° about x
        y_axis = np.array([0, 1, 0])
        np.testing.assert_allclose(R @ y_axis, [0, 0, 1], atol=1e-10)

    def test_rotation_about_y(self) -> None:
        """Test rotation about y-axis."""
        T = Transform6DOF.from_rotation_y(np.pi / 2)
        R = T.rotation_matrix

        # z -> x for 90° about y
        z_axis = np.array([0, 0, 1])
        np.testing.assert_allclose(R @ z_axis, [1, 0, 0], atol=1e-10)

    def test_rotation_about_z(self) -> None:
        """Test rotation about z-axis."""
        T = Transform6DOF.from_rotation_z(np.pi / 2)
        R = T.rotation_matrix

        # x -> y for 90° about z
        x_axis = np.array([1, 0, 0])
        np.testing.assert_allclose(R @ x_axis, [0, 1, 0], atol=1e-10)

    def test_rotation_about_arbitrary_axis(self) -> None:
        """Test rotation about arbitrary axis."""
        # 180° rotation about axis [1, 1, 0] (normalized)
        axis = np.array([1, 1, 0]) / np.sqrt(2)
        T = Transform6DOF.from_axis_angle(axis, np.pi)
        R = T.rotation_matrix

        # z should be flipped
        z_axis = np.array([0, 0, 1])
        np.testing.assert_allclose(R @ z_axis, [0, 0, -1], atol=1e-10)

    def test_from_rotation_matrix(self) -> None:
        """Test creating transform from rotation matrix."""
        # 90° about z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        T = Transform6DOF.from_rotation_matrix(R, [1, 2, 3])
        np.testing.assert_allclose(T.rotation_matrix, R, atol=1e-10)
        np.testing.assert_allclose(T.translation, [1, 2, 3], atol=1e-10)

    def test_compose_transforms(self) -> None:
        """Test composing two transforms."""
        T1 = Transform6DOF.from_translation([1, 0, 0])
        T2 = Transform6DOF.from_translation([0, 1, 0])
        T3 = T1.compose(T2)
        np.testing.assert_allclose(T3.translation, [1, 1, 0], atol=1e-10)

    def test_inverse_transform(self) -> None:
        """Test transform inversion."""
        T = Transform6DOF.from_rotation_z(np.pi / 4)
        T = T.compose(Transform6DOF.from_translation([1, 2, 3]))
        T_inv = T.inverse()

        # T * T_inv should be identity
        identity = T.compose(T_inv)
        np.testing.assert_allclose(identity.translation, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(identity.rotation_matrix, np.eye(3), atol=1e-10)

    def test_transform_point(self) -> None:
        """Test transforming a point."""
        T = Transform6DOF.from_rotation_z(np.pi / 2)
        T = T.compose(Transform6DOF.from_translation([1, 0, 0]))

        point = np.array([1, 0, 0])
        transformed = T.transform_point(point)
        # Rotate [1,0,0] -> [0,1,0], then translate by [1,0,0]
        np.testing.assert_allclose(transformed, [1, 1, 0], atol=1e-10)

    def test_transform_points_batch(self) -> None:
        """Test transforming multiple points."""
        T = Transform6DOF.from_translation([1, 0, 0])
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        transformed = T.transform_points(points)
        expected = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]])
        np.testing.assert_allclose(transformed, expected, atol=1e-10)

    def test_to_homogeneous_matrix(self) -> None:
        """Test conversion to 4x4 homogeneous matrix."""
        T = Transform6DOF.from_translation([1, 2, 3])
        T = T.compose(Transform6DOF.from_rotation_z(np.pi / 2))
        H = T.homogeneous_matrix
        assert H.shape == (4, 4)
        np.testing.assert_allclose(H[3, :], [0, 0, 0, 1], atol=1e-10)

    def test_from_homogeneous_matrix(self) -> None:
        """Test creating transform from 4x4 matrix."""
        H = np.eye(4)
        H[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90° about z
        H[:3, 3] = [1, 2, 3]

        T = Transform6DOF.from_homogeneous_matrix(H)
        np.testing.assert_allclose(T.translation, [1, 2, 3], atol=1e-10)

        x_axis = np.array([1, 0, 0])
        np.testing.assert_allclose(T.rotation_matrix @ x_axis, [0, 1, 0], atol=1e-10)

    def test_to_spatial_transform_6x6(self) -> None:
        """Test conversion to 6x6 Plücker transform."""
        T = Transform6DOF.from_translation([1, 2, 3])
        X = T.to_spatial_transform()
        assert X.shape == (6, 6)

    def test_interpolate_transforms(self) -> None:
        """Test linear interpolation between transforms."""
        T1 = Transform6DOF.identity()
        T2 = Transform6DOF.from_translation([2, 0, 0])

        T_mid = Transform6DOF.interpolate(T1, T2, 0.5)
        np.testing.assert_allclose(T_mid.translation, [1, 0, 0], atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
