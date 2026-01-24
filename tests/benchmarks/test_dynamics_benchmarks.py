"""Benchmark tests for physics dynamics."""

import importlib.util

import numpy as np
import pytest

from src.shared.python.constants import GRAVITY_M_S2
from src.shared.python.engine_availability import MUJOCO_AVAILABLE
from src.shared.python.path_utils import setup_import_paths

# Import paths configured at test runner level via pyproject.toml/conftest.py
# This is a fallback for benchmark runners that may not use the full test setup
setup_import_paths()

# Check if pytest-benchmark is installed, otherwise skip
if importlib.util.find_spec("pytest_benchmark") is None:
    pytest.skip("pytest-benchmark not installed", allow_module_level=True)

if MUJOCO_AVAILABLE:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.aba import (
        aba,
    )
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.crba import (
        crba,
    )
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.rnea import (
        rnea,
    )
else:
    pytest.skip("MuJoCo dynamics modules not available", allow_module_level=True)


def create_random_model(num_bodies=10):
    """
    Create a random kinematic chain model for benchmarking.
    """
    model = {}
    model["NB"] = num_bodies
    model["parent"] = np.array([-1] + [i - 1 for i in range(1, num_bodies)], dtype=int)
    model["jtype"] = ["Rz"] * num_bodies  # Revolute joints (z-axis)

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

    model["gravity"] = np.array([0, 0, 0, 0, 0, -GRAVITY_M_S2])
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


def test_crba_benchmark(benchmark, dynamics_setup):
    """Benchmark the Composite Rigid Body Algorithm."""
    model, q, _, _, _ = dynamics_setup
    benchmark(crba, model, q)


def test_rnea_benchmark(benchmark, dynamics_setup):
    """Benchmark the Recursive Newton-Euler Algorithm."""
    model, q, qd, qdd, _ = dynamics_setup
    benchmark(rnea, model, q, qd, qdd)
