"""Unit tests for RNEA module."""

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.rnea import (
    rnea,
)


def test_rnea_1dof_robot():
    """Test RNEA with a simple 1-DOF robot."""
    nb = 1
    model = {
        "NB": nb,
        "parent": np.array([-1]),
        "jtype": ["Rz"],
        "Xtree": [np.eye(6)],
        "I": [np.eye(6)],  # Unit inertia
        "gravity": np.zeros(6),
    }
    q = np.array([0.0])
    qd = np.array([0.0])
    qdd = np.array([1.0])

    tau = rnea(model, q, qd, qdd)

    # 1 DOF, unit inertia, no gravity, qdd=1.0 -> tau = I * qdd = 1.0 * 1.0 = 1.0
    assert tau.shape == (1,)
    np.testing.assert_array_almost_equal(tau, np.array([1.0]))


def test_rnea_1dof_with_gravity():
    """Test RNEA with a simple 1-DOF robot and gravity."""
    nb = 1
    # Rotate around x, so z-gravity creates torque
    model = {
        "NB": nb,
        "parent": np.array([-1]),
        "jtype": ["Rz"],  # pure rotation
        "Xtree": [np.eye(6)],
        "I": [np.eye(6)],  # Unit inertia
        "gravity": np.array([0, 0, 0, 0, 0, -9.81]),
    }
    q = np.array([0.0])
    qd = np.array([0.0])
    qdd = np.array([1.0])

    tau = rnea(model, q, qd, qdd)

    assert tau.shape == (1,)
    # For Rz at origin, gravity is along z, rotation is around z, gravity doesn't affect it.
    np.testing.assert_array_almost_equal(tau, np.array([1.0]))


def test_rnea_invalid_inputs():
    """Test RNEA with invalid inputs."""
    model = {
        "NB": 2,
        "parent": np.array([-1, 0]),
        "jtype": ["Rz", "Rz"],
        "Xtree": [np.eye(6), np.eye(6)],
        "I": [np.eye(6), np.eye(6)],
    }

    with pytest.raises(ValueError, match="q must have length 2"):
        rnea(model, np.array([0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))

    with pytest.raises(ValueError, match="qd must have length 2"):
        rnea(model, np.array([0.0, 0.0]), np.array([0.0]), np.array([0.0, 0.0]))

    with pytest.raises(ValueError, match="qdd must have length 2"):
        rnea(model, np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0]))
