# ruff: noqa: E402
"""Unit tests for OpenSim Physics Engine."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Explicit attribute lists for OpenSim C++ types (opensim-core bindings).
_OSIM_MODEL_SPEC = [
    "getName",
    "initSystem",
    "initializeState",
    "equilibrateMuscles",
    "getNumCoordinates",
    "getNumSpeeds",
    "getMatterSubsystem",
    "realizeVelocity",
    "realizePosition",
]
_OSIM_STATE_SPEC = [
    "getQ",
    "getU",
    "getTime",
    "setQ",
    "setU",
]
_OSIM_MANAGER_SPEC = [
    "integrate",
    "setSessionTime",
    "setIntegrator",
    "setInitialTime",
    "setFinalTime",
]

# Mock opensim before importing the engine
mock_opensim = MagicMock()
sys.modules["opensim"] = mock_opensim

# Also patch OPENSIM_AVAILABLE so the engine module picks up the mock
with patch("src.shared.python.engine_core.engine_availability.OPENSIM_AVAILABLE", True):
    from src.engines.physics_engines.opensim.python import (  # noqa: E402
        opensim_physics_engine as osim_module,
    )
    from src.engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E402
        OpenSimPhysicsEngine,
    )

    # Force the module-level opensim reference to use our mock
    osim_module.opensim = mock_opensim


def teardown_module(module):
    """Clean up sys.modules pollution."""
    if "opensim" in sys.modules:
        del sys.modules["opensim"]


@pytest.fixture
def engine():
    # Reset mock_opensim for each test
    mock_opensim.reset_mock()
    # Ensure the engine module's opensim ref points to our mock
    osim_module.opensim = mock_opensim
    return OpenSimPhysicsEngine()


def test_initialization(engine):
    assert engine.model_name == "OpenSim_NoModel"


def test_load_from_path(engine):
    path = "test_model.osim"

    mock_model = MagicMock(spec=_OSIM_MODEL_SPEC)
    mock_model.getName.return_value = "TestModel"
    mock_opensim.Model.return_value = mock_model

    with patch("os.path.exists", return_value=True):
        engine.load_from_path(path)

    mock_opensim.Model.assert_called_with(path)
    mock_model.initSystem.assert_called_once()
    assert engine.model_name == "TestModel"


@patch("tempfile.NamedTemporaryFile")
def test_load_from_string(mock_named_temp, engine):
    # Setup mock temp file
    mock_tmp = MagicMock(spec=["name", "write", "flush"])
    mock_tmp.name = "/tmp/fake.osim"
    # context manager return
    mock_named_temp.return_value.__enter__.return_value = mock_tmp

    # Mock load_from_path to avoid real loading logic
    with patch.object(engine, "load_from_path") as mock_load:
        engine.load_from_string("<osim/>")
        mock_load.assert_called_once_with("/tmp/fake.osim")

    # Check that write was called
    mock_tmp.write.assert_called_once_with("<osim/>")


def test_reset(engine):
    # Setup loaded model
    engine._model = MagicMock(spec=_OSIM_MODEL_SPEC)
    engine._state = MagicMock(spec=_OSIM_STATE_SPEC)
    engine._manager = MagicMock(spec=_OSIM_MANAGER_SPEC)

    engine.reset()

    engine._model.initializeState.assert_called_once()
    engine._model.equilibrateMuscles.assert_called_once()
    engine._manager.setSessionTime.assert_called_with(0.0)


def test_step(engine):
    # Setup loaded model
    engine._model = MagicMock(spec=_OSIM_MODEL_SPEC)
    engine._state = MagicMock(spec=_OSIM_STATE_SPEC)
    engine._manager = MagicMock(spec=_OSIM_MANAGER_SPEC)

    # Mock current time
    engine._state.getTime.return_value = 1.0

    engine.step(0.01)

    engine._manager.integrate.assert_called_with(1.01)


def test_get_state(engine):
    engine._model = MagicMock(spec=_OSIM_MODEL_SPEC)
    engine._state = MagicMock(spec=_OSIM_STATE_SPEC)

    # Mock sizes
    engine._model.getNumCoordinates.return_value = 2
    engine._model.getNumSpeeds.return_value = 2

    # Mock vectors
    mock_q = MagicMock(spec=["get"])
    mock_q.get.side_effect = [0.1, 0.2]
    engine._state.getQ.return_value = mock_q

    mock_u = MagicMock(spec=["get"])
    mock_u.get.side_effect = [0.01, 0.02]
    engine._state.getU.return_value = mock_u

    q, v = engine.get_state()

    assert np.allclose(q, [0.1, 0.2])
    assert np.allclose(v, [0.01, 0.02])


def test_set_state(engine):
    engine._model = MagicMock(spec=_OSIM_MODEL_SPEC)
    engine._state = MagicMock(spec=_OSIM_STATE_SPEC)

    engine._model.getNumCoordinates.return_value = 2
    engine._model.getNumSpeeds.return_value = 2

    q = np.array([0.1, 0.2])
    v = np.array([0.01, 0.02])

    engine.set_state(q, v)

    engine._state.setQ.assert_called()
    engine._state.setU.assert_called()
    engine._model.realizeVelocity.assert_called_with(engine._state)


def test_compute_mass_matrix(engine):
    engine._model = MagicMock(spec=_OSIM_MODEL_SPEC)
    engine._state = MagicMock(spec=_OSIM_STATE_SPEC)

    engine._model.getNumSpeeds.return_value = 2

    mock_matter = MagicMock(spec=["calcM"])
    engine._model.getMatterSubsystem.return_value = mock_matter

    # Mock matrix behavior
    mock_matrix = MagicMock(spec=["get"])
    mock_matrix.get.return_value = 1.0
    mock_opensim.Matrix.return_value = mock_matrix

    M = engine.compute_mass_matrix()

    mock_matter.calcM.assert_called()
    assert M.shape == (2, 2)
