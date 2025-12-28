"""Unit tests for MuJoCoPhysicsEngine."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import sys

# Mock mujoco before importing the engine if it's imported at top level
sys.modules["mujoco"] = MagicMock()
from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import MuJoCoPhysicsEngine

@pytest.fixture
def engine():
    return MuJoCoPhysicsEngine()

def test_initialization(engine):
    assert engine.model is None
    assert engine.data is None

@patch("engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.mujoco")
def test_load_from_xml_string(mock_mujoco, engine):
    xml = "<mujoco/>"
    engine.load_from_xml_string(xml)
    
    mock_mujoco.MjModel.from_xml_string.assert_called_once_with(xml)
    assert engine.model is not None
    assert engine.data is not None

@patch("engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.mujoco")
def test_load_from_path(mock_mujoco, engine):
    path = "model.xml"
    engine.load_from_path(path)
    
    mock_mujoco.MjModel.from_xml_path.assert_called_once_with(path)
    assert engine.xml_path == path

@patch("engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.mujoco")
def test_step(mock_mujoco, engine):
    # Setup mock model/data
    engine.model = MagicMock()
    engine.data = MagicMock()
    
    engine.step()
    
    mock_mujoco.mj_step.assert_called_once_with(engine.model, engine.data)

@patch("engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.mujoco")
def test_reset(mock_mujoco, engine):
    # Setup mock model/data
    engine.model = MagicMock()
    engine.data = MagicMock()
    
    engine.reset()
    
    mock_mujoco.mj_resetData.assert_called_once_with(engine.model, engine.data)
    mock_mujoco.mj_forward.assert_called_once()  # called by forward()

def test_set_control(engine):
    engine.model = MagicMock()
    engine.model.nu = 2
    engine.data = MagicMock()
    engine.data.ctrl = np.zeros(2)
    
    ctrl = np.array([1.0, 2.0])
    engine.set_control(ctrl)
    
    np.testing.assert_array_equal(engine.data.ctrl, ctrl)

def test_set_control_mismatch(engine):
    engine.model = MagicMock()
    engine.model.nu = 2
    engine.data = MagicMock()
    
    ctrl = np.array([1.0, 2.0, 3.0])
    # Should warn but not raise
    engine.set_control(ctrl)
