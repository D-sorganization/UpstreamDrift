"""Unit tests for OpenSim Physics Engine stub."""

import numpy as np
import pytest

from engines.physics_engines.opensim.python.opensim_physics_engine import (
    OpenSimPhysicsEngine,
)


@pytest.fixture
def engine():
    return OpenSimPhysicsEngine()


def test_initialization(engine):
    assert engine.model_name == "OpenSimStub"


def test_methods_raise_not_implemented(engine):
    with pytest.raises(NotImplementedError):
        engine.load_from_path("dummy.osim")

    with pytest.raises(NotImplementedError):
        engine.load_from_string("<osim/>")


def test_stubs_return_safe_defaults(engine):
    # Ensure stubs that return values don't crash and return correct types/shapes
    engine.reset()
    engine.step(0.01)
    engine.forward()
    engine.set_control(np.array([1.0]))
    engine.set_state(np.array([1.0]), np.array([1.0]))

    q, v = engine.get_state()
    assert isinstance(q, np.ndarray) and q.size == 0
    assert isinstance(v, np.ndarray) and v.size == 0

    assert engine.get_time() == 0.0

    M = engine.compute_mass_matrix()
    assert isinstance(M, np.ndarray)

    tau = engine.compute_inverse_dynamics(np.array([]))
    assert isinstance(tau, np.ndarray)

    J = engine.compute_jacobian("body")
    assert J is None
