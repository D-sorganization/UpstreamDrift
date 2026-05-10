"""Unit tests for CRBA module."""

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.crba import (
    crba,
)


def test_crba_1dof_robot():
    """Test CRBA with a simple 1-DOF robot."""
    nb = 1
    model = {
        "NB": nb,
        "parent": np.array([-1]),
        "jtype": ["Rz"],
        "Xtree": [np.eye(6)],
        "I": [np.eye(6)],  # Unit inertia
    }
    q = np.array([0.0])

    H = crba(model, q)

    assert H.shape == (1, 1)
    # Rz joint selects z-axis angular inertia which is 1
    np.testing.assert_array_almost_equal(H, np.array([[1.0]]))


def test_crba_2dof_robot():
    """Test CRBA with a simple 2-DOF robot."""
    nb = 2
    model = {
        "NB": nb,
        "parent": np.array([-1, 0]),
        "jtype": ["Rz", "Rz"],
        "Xtree": [np.eye(6), np.eye(6)],  # No offsets
        "I": [np.eye(6), np.eye(6)],  # Unit inertias
    }
    q = np.array([0.0, 0.0])

    H = crba(model, q)

    assert H.shape == (2, 2)
    # Both are Rz joints, no offsets, so inertias add up for the parent
    # Parent inertia = I1 + I2 = 2. Children inertia = 1
    # Coupling = 1
    expected = np.array([[2.0, 1.0], [1.0, 1.0]])
    np.testing.assert_array_almost_equal(H, expected)


def test_crba_invalid_q():
    """Test CRBA with invalid q vector."""
    model = {
        "NB": 2,
        "parent": np.array([-1, 0]),
        "jtype": ["Rz", "Rz"],
        "Xtree": [np.eye(6), np.eye(6)],
        "I": [np.eye(6), np.eye(6)],
    }

    with pytest.raises(ValueError, match="q must have length 2"):
        crba(model, np.array([0.0]))
