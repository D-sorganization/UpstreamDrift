"""Unit tests for twists and wrenches representations."""

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.screw_theory.twists import (
    twist_to_spatial,
    wrench_to_spatial,
)


def test_twist_to_spatial_origin():
    """Test twist to spatial vector at origin."""
    omega = np.array([0, 0, 1], dtype=float)
    v = np.array([1, 0, 0], dtype=float)

    V = twist_to_spatial(omega, v)

    expected = np.array([0, 0, 1, 1, 0, 0], dtype=float)
    np.testing.assert_array_almost_equal(V, expected)


def test_twist_to_spatial_with_point():
    """Test twist to spatial vector with a reference point."""
    omega = np.array([0, 0, 1], dtype=float)
    v = np.array([0, 0, 0], dtype=float)
    point = np.array([1, 0, 0], dtype=float)

    # v_new = v - omega x point = [0,0,0] - [0,0,1] x [1,0,0] = [0,-1,0]
    V = twist_to_spatial(omega, v, point)

    expected = np.array([0, 0, 1, 0, -1, 0], dtype=float)
    np.testing.assert_array_almost_equal(V, expected)


def test_twist_to_spatial_invalid_shape():
    """Test twist to spatial vector with invalid shapes."""
    with pytest.raises(ValueError):
        twist_to_spatial(np.array([1, 0]), np.array([0, 0, 0]))

    with pytest.raises(ValueError):
        twist_to_spatial(np.array([1, 0, 0]), np.array([0, 0]))

    with pytest.raises(ValueError):
        twist_to_spatial(
            np.array([1, 0, 0]), np.array([0, 0, 0]), point=np.array([1, 0])
        )


def test_wrench_to_spatial_origin():
    """Test wrench to spatial vector at origin."""
    moment = np.array([0, 0, 1], dtype=float)
    force = np.array([10, 0, 0], dtype=float)

    F = wrench_to_spatial(moment, force)

    expected = np.array([0, 0, 1, 10, 0, 0], dtype=float)
    np.testing.assert_array_almost_equal(F, expected)


def test_wrench_to_spatial_with_point():
    """Test wrench to spatial vector with a reference point."""
    moment = np.array([0, 0, 0], dtype=float)
    force = np.array([10, 0, 0], dtype=float)
    point = np.array([0, 1, 0], dtype=float)

    # moment_new = moment + point x force = [0,0,0] + [0,1,0] x [10,0,0] = [0,0,-10]
    F = wrench_to_spatial(moment, force, point)

    expected = np.array([0, 0, -10, 10, 0, 0], dtype=float)
    np.testing.assert_array_almost_equal(F, expected)


def test_wrench_to_spatial_invalid_shape():
    """Test wrench to spatial vector with invalid shapes."""
    with pytest.raises(ValueError):
        wrench_to_spatial(np.array([1, 0]), np.array([0, 0, 0]))

    with pytest.raises(ValueError):
        wrench_to_spatial(np.array([1, 0, 0]), np.array([0, 0]))

    with pytest.raises(ValueError):
        wrench_to_spatial(
            np.array([1, 0, 0]), np.array([0, 0, 0]), point=np.array([1, 0])
        )
