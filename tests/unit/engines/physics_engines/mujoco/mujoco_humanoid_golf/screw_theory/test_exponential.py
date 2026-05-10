"""Unit tests for exponential map and logarithmic map."""

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.screw_theory.exponential import (
    exponential_map,
    logarithmic_map,
)


def test_exponential_map_pure_translation():
    """Test exponential map with pure translation."""
    S = np.array([0, 0, 0, 1, 0, 0], dtype=float)
    T = exponential_map(S, 2.0)

    expected = np.eye(4)
    expected[0, 3] = 2.0

    np.testing.assert_array_almost_equal(T, expected)


def test_exponential_map_pure_rotation():
    """Test exponential map with pure rotation."""
    S = np.array([0, 0, 1, 0, 0, 0], dtype=float)
    T = exponential_map(S, np.pi / 2)

    expected = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )

    np.testing.assert_array_almost_equal(T, expected)


def test_exponential_map_invalid_shape():
    """Test exponential map with invalid shape."""
    with pytest.raises(AssertionError):
        exponential_map(np.zeros(5), 1.0)


def test_logarithmic_map_identity():
    """Test logarithmic map with identity matrix."""
    T = np.eye(4)
    S, theta = logarithmic_map(T)

    assert theta == 0.0
    np.testing.assert_array_almost_equal(S, np.zeros(6))


def test_logarithmic_map_pure_translation():
    """Test logarithmic map with pure translation."""
    T = np.eye(4)
    T[1, 3] = 3.0

    S, theta = logarithmic_map(T)

    assert theta == 3.0
    expected_S = np.array([0, 0, 0, 0, 1, 0], dtype=float)
    np.testing.assert_array_almost_equal(S, expected_S)


def test_logarithmic_map_pure_rotation():
    """Test logarithmic map with pure rotation."""
    T = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    S, theta = logarithmic_map(T)

    np.testing.assert_almost_equal(theta, np.pi / 2)
    expected_S = np.array([0, 0, 1, 0, 0, 0], dtype=float)
    np.testing.assert_array_almost_equal(S, expected_S)


def test_logarithmic_map_180_rotation():
    """Test logarithmic map with 180 degree rotation."""
    # Rotate 180 around X
    T = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float
    )

    S, theta = logarithmic_map(T)

    np.testing.assert_almost_equal(theta, np.pi)
    expected_S = np.array([1, 0, 0, 0, 0, 0], dtype=float)
    # The sign of the axis could be flipped for 180, so we check absolute
    np.testing.assert_array_almost_equal(np.abs(S), np.abs(expected_S))


def test_exponential_logarithmic_roundtrip():
    """Test roundtrip conversion."""
    S = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    S = S / np.linalg.norm(S[:3])  # Normalize omega
    theta = 1.5

    T = exponential_map(S, theta)
    S_out, theta_out = logarithmic_map(T)

    np.testing.assert_almost_equal(theta, theta_out)
    np.testing.assert_array_almost_equal(S, S_out)


def test_logarithmic_map_invalid_shape():
    """Test logarithmic map with invalid shape."""
    T = np.eye(3)
    with pytest.raises(AssertionError):
        logarithmic_map(T)
