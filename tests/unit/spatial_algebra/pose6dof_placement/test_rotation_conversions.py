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


class TestRotationConversions:
    """Tests for rotation representation conversions."""

    def test_euler_to_quaternion(self) -> None:
        """Test euler to quaternion conversion."""
        euler = [0, 0, np.pi / 2]  # 90° yaw
        quat = euler_to_quaternion(euler)

        # Quaternion should have unit norm
        assert np.linalg.norm(quat) == pytest.approx(1.0)

        # Roundtrip
        euler_back = quaternion_to_euler(quat)
        np.testing.assert_allclose(euler, euler_back, atol=1e-10)

    def test_euler_to_rotation_matrix(self) -> None:
        """Test euler to rotation matrix conversion."""
        # 90° about z
        euler = [0, 0, np.pi / 2]
        R = euler_to_rotation_matrix(euler)

        x = np.array([1, 0, 0])
        np.testing.assert_allclose(R @ x, [0, 1, 0], atol=1e-10)

    def test_rotation_matrix_to_euler(self) -> None:
        """Test rotation matrix to euler conversion."""
        euler_orig = [0.1, 0.2, 0.3]
        R = euler_to_rotation_matrix(euler_orig)
        euler_back = rotation_matrix_to_euler(R)
        np.testing.assert_allclose(euler_orig, euler_back, atol=1e-10)

    def test_axis_angle_to_rotation_matrix(self) -> None:
        """Test axis-angle to rotation matrix conversion."""
        # 90° about z
        R = axis_angle_to_rotation_matrix([0, 0, 1], np.pi / 2)
        x = np.array([1, 0, 0])
        np.testing.assert_allclose(R @ x, [0, 1, 0], atol=1e-10)

    def test_quaternion_multiply(self) -> None:
        """Test quaternion multiplication."""
        # Two 45° rotations about z should equal 90°
        q1 = euler_to_quaternion([0, 0, np.pi / 4])
        q2 = euler_to_quaternion([0, 0, np.pi / 4])
        q3 = quaternion_multiply(q1, q2)

        euler = quaternion_to_euler(q3)
        assert euler[2] == pytest.approx(np.pi / 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
