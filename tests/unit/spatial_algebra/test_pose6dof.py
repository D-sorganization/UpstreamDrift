"""Tests for src.shared.python.spatial_algebra.pose6dof module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.spatial_algebra.pose6dof import (
    Pose6DOF,
    Transform6DOF,
    axis_angle_to_rotation_matrix,
    euler_to_quaternion,
    euler_to_rotation_matrix,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
    slerp,
)


class TestEulerToRotationMatrix:
    """Tests for euler_to_rotation_matrix function."""

    def test_identity(self) -> None:
        R = euler_to_rotation_matrix([0, 0, 0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_90_yaw(self) -> None:
        R = euler_to_rotation_matrix([0, 0, math.pi / 2])
        # Yaw = 90°: x→-y, y→x
        expected_x = R @ np.array([1, 0, 0])
        assert expected_x[1] == pytest.approx(1.0, abs=1e-10)

    def test_roundtrip_euler(self) -> None:
        euler_in = [0.1, 0.2, 0.3]
        R = euler_to_rotation_matrix(euler_in)
        euler_out = rotation_matrix_to_euler(R)
        np.testing.assert_allclose(euler_out, euler_in, atol=1e-10)


class TestQuaternionConversions:
    """Tests for quaternion conversion functions."""

    def test_identity_euler_to_quat(self) -> None:
        q = euler_to_quaternion([0, 0, 0])
        assert q[0] == pytest.approx(1.0, abs=1e-10)

    def test_quat_euler_roundtrip(self) -> None:
        euler_in = [0.1, 0.2, 0.3]
        q = euler_to_quaternion(euler_in)
        euler_out = quaternion_to_euler(q)
        np.testing.assert_allclose(euler_out, euler_in, atol=1e-10)

    def test_quat_to_rotation_matrix_and_back(self) -> None:
        q = euler_to_quaternion([0.3, -0.2, 0.1])
        R = quaternion_to_rotation_matrix(q)
        q_back = rotation_matrix_to_quaternion(R)
        # quaternions are unique up to sign
        if q[0] * q_back[0] < 0:
            q_back = -q_back
        np.testing.assert_allclose(q_back, q, atol=1e-10)

    def test_rotation_matrix_to_quat_identity(self) -> None:
        q = rotation_matrix_to_quaternion(np.eye(3))
        assert abs(q[0]) == pytest.approx(1.0, abs=1e-10)


class TestQuaternionOperations:
    """Tests for quaternion multiply, inverse, slerp."""

    def test_multiply_identity(self) -> None:
        q = np.array([1.0, 0, 0, 0])
        result = quaternion_multiply(q, q)
        np.testing.assert_allclose(result, q, atol=1e-12)

    def test_inverse(self) -> None:
        q = euler_to_quaternion([0.5, 0.3, 0.1])
        q_inv = quaternion_inverse(q)
        product = quaternion_multiply(q, q_inv)
        np.testing.assert_allclose(abs(product[0]), 1.0, atol=1e-10)

    def test_slerp_endpoints(self) -> None:
        q1 = np.array([1.0, 0, 0, 0])
        q2 = euler_to_quaternion([0.0, 0.0, math.pi / 2])
        result_0 = slerp(q1, q2, 0.0)
        result_1 = slerp(q1, q2, 1.0)
        if q1[0] * result_0[0] < 0:
            result_0 = -result_0
        np.testing.assert_allclose(result_0, q1, atol=1e-10)
        if q2[0] * result_1[0] < 0:
            result_1 = -result_1
        np.testing.assert_allclose(result_1, q2, atol=1e-10)

    def test_slerp_midpoint(self) -> None:
        q1 = np.array([1.0, 0, 0, 0])
        q2 = euler_to_quaternion([0.0, 0.0, math.pi / 2])
        mid = slerp(q1, q2, 0.5)
        assert np.linalg.norm(mid) == pytest.approx(1.0, rel=1e-10)


class TestAxisAngle:
    """Tests for axis-angle to rotation matrix."""

    def test_zero_angle(self) -> None:
        R = axis_angle_to_rotation_matrix([0, 0, 1], 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_180_about_z(self) -> None:
        R = axis_angle_to_rotation_matrix([0, 0, 1], math.pi)
        # x→-x, y→-y
        assert R[0, 0] == pytest.approx(-1.0, abs=1e-10)
        assert R[1, 1] == pytest.approx(-1.0, abs=1e-10)
        assert R[2, 2] == pytest.approx(1.0, abs=1e-10)


class TestPose6DOF:
    """Tests for Pose6DOF class."""

    def test_default_pose(self) -> None:
        p = Pose6DOF()
        np.testing.assert_allclose(p.position, [0, 0, 0])
        np.testing.assert_allclose(p.euler_angles, [0, 0, 0])

    def test_custom_position(self) -> None:
        p = Pose6DOF(position=[1, 2, 3])
        assert p.x == pytest.approx(1.0)
        assert p.y == pytest.approx(2.0)
        assert p.z == pytest.approx(3.0)

    def test_set_position(self) -> None:
        p = Pose6DOF()
        p.position = [5, 6, 7]
        assert p.x == pytest.approx(5.0)

    def test_from_quaternion(self) -> None:
        p = Pose6DOF.from_quaternion([1, 2, 3], [1, 0, 0, 0])
        assert p.x == pytest.approx(1.0)

    def test_from_rotation_matrix(self) -> None:
        p = Pose6DOF.from_rotation_matrix([0, 0, 0], np.eye(3))
        np.testing.assert_allclose(p.euler_angles, [0, 0, 0], atol=1e-10)

    def test_to_quaternion(self) -> None:
        p = Pose6DOF(euler_angles=[0, 0, math.pi / 2])
        q = p.to_quaternion()
        assert abs(q[0]) == pytest.approx(math.cos(math.pi / 4), abs=1e-6)

    def test_rotation_matrix_property(self) -> None:
        p = Pose6DOF()
        R = p.rotation_matrix
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_copy(self) -> None:
        p = Pose6DOF(position=[1, 2, 3], euler_angles=[0.1, 0.2, 0.3])
        c = p.copy()
        assert c == p
        c.x = 999
        assert p.x != 999  # deep copy

    def test_eq(self) -> None:
        a = Pose6DOF(position=[1, 2, 3])
        b = Pose6DOF(position=[1, 2, 3])
        assert a == b

    def test_repr(self) -> None:
        p = Pose6DOF()
        assert "Pose6DOF" in repr(p)

    def test_translate_returns_new(self) -> None:
        """translate() returns a new Pose6DOF (not in-place)."""
        p = Pose6DOF(position=[1, 0, 0])
        p2 = p.translate([1, 0, 0])
        assert p2.x == pytest.approx(2.0)
        assert p.x == pytest.approx(1.0)  # original unchanged

    def test_homogeneous_matrix(self) -> None:
        p = Pose6DOF(position=[1, 2, 3])
        H = p.homogeneous_matrix
        assert H.shape == (4, 4)
        assert H[0, 3] == pytest.approx(1.0)
        assert H[3, 3] == pytest.approx(1.0)


class TestTransform6DOF:
    """Tests for Transform6DOF class."""

    def test_default_transform(self) -> None:
        t = Transform6DOF()
        H = t.homogeneous_matrix
        np.testing.assert_allclose(H, np.eye(4), atol=1e-15)

    def test_compose_with_identity(self) -> None:
        t1 = Transform6DOF(translation=[1, 2, 3])
        t_id = Transform6DOF()
        result = t1.compose(t_id)
        np.testing.assert_allclose(result.translation, [1, 2, 3], atol=1e-10)

    def test_inverse(self) -> None:
        t = Transform6DOF(translation=[1, 2, 3])
        t_inv = t.inverse()
        composed = t.compose(t_inv)
        np.testing.assert_allclose(composed.homogeneous_matrix, np.eye(4), atol=1e-10)
