"""Unit tests for CRBA algorithm."""

import numpy as np
import pytest

from src.shared.python.constants import GRAVITY_M_S2
from src.shared.python.engine_availability import MUJOCO_AVAILABLE, skip_if_unavailable

pytestmark = skip_if_unavailable("mujoco")

if MUJOCO_AVAILABLE:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.crba import (
        crba,
    )


def create_random_model(num_bodies=5):
    """
    Create a random kinematic chain model for testing.
    """
    model = {}
    model["NB"] = num_bodies
    model["parent"] = np.array([-1] + [i - 1 for i in range(1, num_bodies)], dtype=int)
    model["jtype"] = ["Rx"] * num_bodies  # Revolute joints (x-axis)

    # Random transforms
    model["Xtree"] = [np.eye(6) for _ in range(num_bodies)]

    # Random inertias (should be positive definite)
    model["I"] = []
    for _ in range(num_bodies):
        # Create random spatial inertia
        # mass = 1, diagonal inertia
        mass = 1.0
        I_3x3 = np.eye(3)
        # Construct 6x6 spatial inertia
        spatial_inertia = np.zeros((6, 6))
        spatial_inertia[:3, :3] = I_3x3
        spatial_inertia[3:, 3:] = mass * np.eye(3)
        model["I"].append(spatial_inertia)

    model["gravity"] = np.array([0, 0, 0, 0, 0, -GRAVITY_M_S2])
    return model


def test_crba_symmetry():
    """Test that CRBA produces a symmetric mass matrix."""
    model = create_random_model(10)
    q = np.random.rand(10)

    H = crba(model, q)

    assert H.shape == (10, 10)
    assert np.allclose(H, H.T), "Mass matrix must be symmetric"


def test_crba_positive_definite():
    """Test that CRBA produces a positive definite mass matrix."""
    model = create_random_model(5)
    q = np.random.rand(5)

    H = crba(model, q)

    # Check eigenvalues are positive
    eigvals = np.linalg.eigvalsh(H)
    assert np.all(eigvals > 0), "Mass matrix must be positive definite"


def test_crba_values():
    """Test CRBA against a simple analytical case (single pendulum)."""
    # Single body (pendulum)
    # Mass m at distance r
    # I = m*r^2
    model = {}
    model["NB"] = 1
    model["parent"] = np.array([-1], dtype=int)  # type: ignore
    model["jtype"] = ["Rx"]  # type: ignore[assignment]
    model["Xtree"] = [np.eye(6)]  # type: ignore[assignment]

    m = 2.0
    r = 0.5
    I_val = m * r**2

    # Spatial inertia for point mass at distance r along Y axis
    # Rotating about X axis
    # For simplicity, let's just set the rotational inertia directly
    # I_xx = I_val
    spatial_inertia = np.zeros((6, 6))
    spatial_inertia[0, 0] = I_val  # Ixx
    spatial_inertia[3, 3] = m  # mass
    spatial_inertia[4, 4] = m
    spatial_inertia[5, 5] = m

    model["I"] = [spatial_inertia]  # type: ignore[assignment]

    q = np.array([0.0])

    H = crba(model, q)

    assert np.isclose(H[0, 0], I_val), f"Expected {I_val}, got {H[0, 0]}"
