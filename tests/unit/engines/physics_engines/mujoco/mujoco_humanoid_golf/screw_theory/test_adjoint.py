"""Unit tests for adjoint transformations."""

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.screw_theory.adjoint import (
    adjoint_transform,
)


def test_adjoint_transform_identity():
    """Test adjoint transform with identity matrix."""
    T = np.eye(4)
    Ad = adjoint_transform(T)

    assert Ad.shape == (6, 6)
    np.testing.assert_array_equal(Ad, np.eye(6))


def test_adjoint_transform_translation_only():
    """Test adjoint transform with pure translation."""
    T = np.eye(4)
    T[0, 3] = 1.0  # translate x by 1
    Ad = adjoint_transform(T)

    expected = np.eye(6)
    expected[3, 1] = -0.0
    expected[3, 2] = 0.0
    expected[4, 0] = 0.0
    expected[4, 2] = -1.0
    expected[5, 0] = -0.0
    expected[5, 1] = 1.0

    np.testing.assert_array_almost_equal(Ad, expected)


def test_adjoint_transform_rotation_only():
    """Test adjoint transform with pure rotation."""
    # Rotate 90 degrees around Z
    T = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    Ad = adjoint_transform(T)

    expected = np.zeros((6, 6))
    expected[:3, :3] = T[:3, :3]
    expected[3:, 3:] = T[:3, :3]

    np.testing.assert_array_almost_equal(Ad, expected)


def test_adjoint_transform_combined():
    """Test adjoint transform with both rotation and translation."""
    T = np.array([[0, -1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    Ad = adjoint_transform(T)

    assert Ad.shape == (6, 6)

    # Check R blocks
    np.testing.assert_array_almost_equal(Ad[:3, :3], T[:3, :3])
    np.testing.assert_array_almost_equal(Ad[3:, 3:], T[:3, :3])

    # Transform twist
    V_b = np.array([0, 0, 1, 0, 0, 0], dtype=float)
    V_a = Ad @ V_b

    # Expected: angular velocity [0, 0, 1], linear velocity is p x w = [1, 0, 0] x [0, 0, 1] = [0, -1, 0]
    expected_V_a = np.array([0, 0, 1, 0, -1, 0], dtype=float)
    np.testing.assert_array_almost_equal(V_a, expected_V_a)


def test_adjoint_transform_invalid_shape():
    """Test adjoint transform with invalid shape."""
    T = np.eye(3)
    with pytest.raises(AssertionError):
        adjoint_transform(T)
