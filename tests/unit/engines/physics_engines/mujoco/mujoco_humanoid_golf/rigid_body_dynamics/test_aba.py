"""Unit tests for ABA module."""

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.aba import (
    aba,
)


def test_aba_1dof_robot():
    """Test ABA with a simple 1-DOF robot."""
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
    tau = np.array([1.0])

    qdd = aba(model, q, qd, tau)

    # 1 DOF, unit inertia, no gravity -> qdd = tau / I = 1.0 / 1.0 = 1.0
    assert qdd.shape == (1,)
    np.testing.assert_array_almost_equal(qdd, np.array([1.0]))


def test_aba_1dof_with_gravity():
    """Test ABA with a simple 1-DOF robot and gravity."""
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
    tau = np.array([1.0])

    qdd = aba(model, q, qd, tau)

    assert qdd.shape == (1,)
    # For Rz at origin, gravity is along z, rotation is around z, gravity doesn't affect it.
    np.testing.assert_array_almost_equal(qdd, np.array([1.0]))


def test_aba_invalid_inputs():
    """Test ABA with invalid inputs."""
    model = {
        "NB": 2,
        "parent": np.array([-1, 0]),
        "jtype": ["Rz", "Rz"],
        "Xtree": [np.eye(6), np.eye(6)],
        "I": [np.eye(6), np.eye(6)],
    }

    with pytest.raises(ValueError, match="q must have length 2"):
        aba(model, np.array([0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))

    with pytest.raises(ValueError, match="qd must have length 2"):
        aba(model, np.array([0.0, 0.0]), np.array([0.0]), np.array([0.0, 0.0]))

    with pytest.raises(ValueError, match="tau must have length 2"):
        aba(model, np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0]))
