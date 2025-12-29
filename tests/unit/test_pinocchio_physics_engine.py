import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock pinocchio before importing
sys.modules["pinocchio"] = MagicMock()

# Mock interfaces
mock_interfaces = MagicMock()
sys.modules["shared.python.interfaces"] = mock_interfaces
class MockPhysicsEngine:
    pass
mock_interfaces.PhysicsEngine = MockPhysicsEngine

from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (  # noqa: E402
    PinocchioPhysicsEngine,
)


@pytest.fixture
def engine():
    return PinocchioPhysicsEngine()

def test_initialization(engine):
    assert engine.model is None
    assert engine.data is None
    assert engine.time == 0.0

@patch("engines.physics_engines.pinocchio.python.pinocchio_physics_engine.pin")
def test_load_from_path(mock_pin, engine):
    engine.load_from_path("test.urdf")

    mock_pin.buildModelFromUrdf.assert_called_once_with("test.urdf")
    mock_pin.neutral.assert_called_once()
    assert engine.model is not None
    assert engine.data is not None

@patch("engines.physics_engines.pinocchio.python.pinocchio_physics_engine.pin")
def test_load_from_string(mock_pin, engine):
    content = "<robot/>"
    mock_model = MagicMock()
    mock_model.nv = 2
    mock_pin.buildModelFromXML.return_value = mock_model

    engine.load_from_string(content, "urdf")

    mock_pin.buildModelFromXML.assert_called_once_with(content)
    assert engine.model is not None

def test_step(engine):
    engine.model = MagicMock()
    engine.data = MagicMock()
    engine.q = np.array([0.0])
    engine.v = np.array([0.0])
    engine.tau = np.array([0.0])

    # Mock aba return
    with patch("engines.physics_engines.pinocchio.python.pinocchio_physics_engine.pin") as mock_pin:
        mock_pin.aba.return_value = np.array([1.0]) # acceleration
        mock_pin.integrate.return_value = np.array([0.1])

        engine.step(0.1)

        mock_pin.aba.assert_called_once()
        mock_pin.integrate.assert_called_once()
        np.testing.assert_array_equal(engine.a, np.array([1.0]))
        # v = v + a*dt = 0 + 1.0*0.1 = 0.1
        np.testing.assert_array_equal(engine.v, np.array([0.1]))

def test_compute_mass_matrix(engine):
    engine.model = MagicMock()
    engine.data = MagicMock()
    # Mock data.M
    engine.data.M = np.array([[1.0, 0.2], [0.0, 2.0]]) # Upper triangular example

    with patch("engines.physics_engines.pinocchio.python.pinocchio_physics_engine.pin") as mock_pin:
        M = engine.compute_mass_matrix()

        mock_pin.crba.assert_called_once()
        # Should be symmetrized
        expected = np.array([[1.0, 0.2], [0.2, 2.0]])
        np.testing.assert_array_almost_equal(M, expected)

def test_compute_jacobian(engine):
    engine.model = MagicMock()
    engine.data = MagicMock()
    engine.model.existBodyName.return_value = True
    engine.model.getFrameId.return_value = 1

    with patch("engines.physics_engines.pinocchio.python.pinocchio_physics_engine.pin") as mock_pin:
        # 6x2 Jacobian
        mock_J = np.zeros((6, 2))
        mock_J[0, 0] = 1.0 # Linear x
        mock_J[5, 1] = 1.0 # Angular z
        mock_pin.getFrameJacobian.return_value = mock_J

        J = engine.compute_jacobian("body")

        assert J is not None
        assert "linear" in J
        assert "angular" in J
        np.testing.assert_array_equal(J["linear"], mock_J[:3, :])
        np.testing.assert_array_equal(J["angular"], mock_J[3:, :])
