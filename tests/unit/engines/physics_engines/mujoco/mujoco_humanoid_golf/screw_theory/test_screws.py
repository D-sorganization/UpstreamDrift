"""Unit tests for screw representations."""

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.screw_theory.screws import (
    screw_axis,
    screw_to_transform,
)


def test_screw_axis_pure_rotation():
    """Test screw axis with pure rotation."""
    axis = np.array([0, 0, 1], dtype=float)
    point = np.array([1, 0, 0], dtype=float)

    # Rotation around z, offset by [1,0,0]
    # v = -omega x point = -[0,0,1] x [1,0,0] = -[0,1,0] = [0,-1,0]
    S = screw_axis(axis, point)

    expected = np.array([0, 0, 1, 0, -1, 0], dtype=float)
    np.testing.assert_array_almost_equal(S, expected)


def test_screw_axis_pure_translation():
    """Test screw axis with pure translation."""
    axis = np.array([1, 0, 0], dtype=float)
    point = np.array([0, 0, 0], dtype=float)

    S = screw_axis(axis, point, pitch=np.inf)

    expected = np.array([0, 0, 0, 1, 0, 0], dtype=float)
    np.testing.assert_array_almost_equal(S, expected)


def test_screw_axis_general_motion():
    """Test screw axis with general motion (pitch)."""
    axis = np.array([0, 0, 1], dtype=float)
    point = np.array([0, 0, 0], dtype=float)
    pitch = 2.0

    S = screw_axis(axis, point, pitch=pitch)

    # omega=[0,0,1], v = [0,0,0] + pitch*[0,0,1] = [0,0,2]
    expected = np.array([0, 0, 1, 0, 0, 2], dtype=float)
    np.testing.assert_array_almost_equal(S, expected)


def test_screw_axis_invalid_shape():
    """Test screw axis with invalid shapes."""
    with pytest.raises(ValueError):
        screw_axis(np.array([1, 0]), np.array([0, 0, 0]))

    with pytest.raises(ValueError):
        screw_axis(np.array([1, 0, 0]), np.array([0, 0]))


def test_screw_to_transform():
    """Test converting screw to transform."""
    axis = np.array([0, 0, 1], dtype=float)
    point = np.array([0, 0, 0], dtype=float)
    pitch = 0.0
    theta = np.pi / 2

    T = screw_to_transform(axis, point, pitch, theta)

    expected = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )

    np.testing.assert_array_almost_equal(T, expected)
