"""
Benchmarks for Rigid Body Dynamics Algorithms (ABA and RNEA).
"""

import numpy as np
import pytest

import sys
from pathlib import Path

# Fix for CI: Add the MuJoCo engine python path to sys.path
# This handles the case where the project is not installed as a package
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MUJOCO_PYTHON_PATH = (
    REPO_ROOT / "engines" / "physics_engines" / "mujoco" / "python"
)
if str(MUJOCO_PYTHON_PATH) not in sys.path:
    sys.path.append(str(MUJOCO_PYTHON_PATH))

from mujoco_humanoid_golf.rigid_body_dynamics.aba import (
    aba,
)
from mujoco_humanoid_golf.rigid_body_dynamics.rnea import (
    rnea,
)


def create_random_model(num_bodies=10):
    """
    Create a random kinematic chain model for benchmarking.
    """
    model = {}
    model["NB"] = num_bodies
    model["parent"] = np.array([-1] + [i - 1 for i in range(1, num_bodies)], dtype=int)
    model["jtype"] = ["R"] * num_bodies  # Revolute joints

    # Random transforms
    model["Xtree"] = [np.eye(6) for _ in range(num_bodies)]

    # Random inertias (should be positive definite)
    model["I"] = []
    for _ in range(num_bodies):
        # Create random spatial inertia
        # Just identity for benchmarking purposes is fine, but let's make it slightly realistic
        # mass = 1, diagonal inertia
        mass = 1.0
        I_3x3 = np.eye(3)
        # Construct 6x6 spatial inertia
        spatial_inertia = np.zeros((6, 6))
        spatial_inertia[:3, :3] = I_3x3
        spatial_inertia[3:, 3:] = mass * np.eye(3)
        model["I"].append(spatial_inertia)

    model["gravity"] = np.array([0, 0, 0, 0, 0, -9.81])
    return model


@pytest.fixture
def dynamics_setup():
    """Setup arrays for dynamics benchmarks."""
    nb = 20  # Reasonable size for a humanoid(-ish) robot
    model = create_random_model(nb)
    q = np.random.rand(nb)
    qd = np.random.rand(nb)
    qdd = np.random.rand(nb)
    tau = np.random.rand(nb)
    return model, q, qd, qdd, tau


def test_aba_benchmark(benchmark, dynamics_setup):
    """Benchmark the Articulated Body Algorithm."""
    model, q, qd, _, tau = dynamics_setup
    benchmark(aba, model, q, qd, tau)


def test_rnea_benchmark(benchmark, dynamics_setup):
    """Benchmark the Recursive Newton-Euler Algorithm."""
    model, q, qd, qdd, _ = dynamics_setup
    benchmark(rnea, model, q, qd, qdd)
