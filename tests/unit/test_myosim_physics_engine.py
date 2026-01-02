# ruff: noqa: E402
"""Unit tests for MyoSim Physics Engine."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock mujoco before importing engine
mock_mujoco = MagicMock()
sys.modules["mujoco"] = mock_mujoco

from engines.physics_engines.myosim.python.myosim_physics_engine import (
    MyoSimPhysicsEngine,  # noqa: E402
)


def teardown_module(module):
    """Clean up sys.modules pollution."""
    if "mujoco" in sys.modules:
        del sys.modules["mujoco"]


@pytest.fixture
def engine():
    mock_mujoco.reset_mock()
    return MyoSimPhysicsEngine()


def test_initialization(engine):
    assert engine.model_name == "MyoSim_NoModel"


def test_load_from_path(engine):
    path = "test_model.xml"
    with patch("os.path.exists", return_value=True):
        mock_model = MagicMock()
        mock_mujoco.MjModel.from_xml_path.return_value = mock_model

        engine.load_from_path(path)

        mock_mujoco.MjModel.from_xml_path.assert_called_with(path)
        assert engine.model_name == "MyoSim Model"


def test_load_from_string(engine):
    content = "<mujoco/>"
    mock_model = MagicMock()
    mock_mujoco.MjModel.from_xml_string.return_value = mock_model

    engine.load_from_string(content)

    mock_mujoco.MjModel.from_xml_string.assert_called_with(content)


def test_step(engine):
    engine.model = MagicMock()
    engine.data = MagicMock()
    # Mock opt.timestep
    engine.model.opt.timestep = 0.01

    engine.step()
    mock_mujoco.mj_step.assert_called_with(engine.model, engine.data)

    # Test with dt override
    engine.step(0.005)
    assert engine.model.opt.timestep == 0.01  # Should be restored


def test_get_state(engine):
    engine.data = MagicMock()
    engine.data.qpos.copy.return_value = np.array([1.0])
    engine.data.qvel.copy.return_value = np.array([0.5])

    q, v = engine.get_state()
    assert np.allclose(q, [1.0])
    assert np.allclose(v, [0.5])


def test_compute_mass_matrix(engine):
    engine.model = MagicMock()
    engine.data = MagicMock()
    engine.model.nv = 2

    M = engine.compute_mass_matrix()

    mock_mujoco.mj_fullM.assert_called()
    assert M.shape == (2, 2)


def test_compute_inverse_dynamics(engine):
    engine.model = MagicMock()
    engine.data = MagicMock()
    engine.model.nv = 2
    engine.data.qfrc_inverse.copy.return_value = np.zeros(2)

    res = engine.compute_inverse_dynamics(np.zeros(2))

    mock_mujoco.mj_inverse.assert_called()
    assert len(res) == 2
