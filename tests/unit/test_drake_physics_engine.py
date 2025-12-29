import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock pydrake before importing
sys.modules["pydrake"] = MagicMock()
sys.modules["pydrake.math"] = MagicMock()
sys.modules["pydrake.multibody"] = MagicMock()
sys.modules["pydrake.multibody.plant"] = MagicMock()
sys.modules["pydrake.multibody.parsing"] = MagicMock()
sys.modules["pydrake.systems"] = MagicMock()
sys.modules["pydrake.systems.framework"] = MagicMock()
sys.modules["pydrake.systems.analysis"] = MagicMock()
sys.modules["pydrake.all"] = MagicMock()

# Mock interfaces
mock_interfaces = MagicMock()
sys.modules["shared.python.interfaces"] = mock_interfaces
class MockPhysicsEngine:
    pass
mock_interfaces.PhysicsEngine = MockPhysicsEngine

from engines.physics_engines.drake.python.drake_physics_engine import (  # noqa: E402
    DrakePhysicsEngine,
)


@pytest.fixture
def engine():
    with patch("engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph") as mock_add:
        mock_plant = MagicMock()
        mock_scene_graph = MagicMock()
        mock_add.return_value = (mock_plant, mock_scene_graph)

        eng = DrakePhysicsEngine()
        eng.plant = mock_plant
        eng.scene_graph = mock_scene_graph
        eng.builder = MagicMock()
        return eng

def test_initialization(engine):
    assert engine.plant is not None
    assert engine.builder is not None
    assert not engine._is_finalized

def test_load_from_path(engine):
    with patch("engines.physics_engines.drake.python.drake_physics_engine.Parser") as mock_parser_cls:
        mock_parser = MagicMock()
        mock_parser_cls.return_value = mock_parser

        path = "test_model.urdf"
        engine.load_from_path(path)

        mock_parser.AddModels.assert_called_once_with(path)
        assert engine.model_name == "test_model"
        # Should ensure finalized
        engine.plant.Finalize.assert_called_once()
        engine.builder.Build.assert_called_once()

def test_load_from_string(engine):
    with patch("engines.physics_engines.drake.python.drake_physics_engine.Parser") as mock_parser_cls:
        mock_parser = MagicMock()
        mock_parser_cls.return_value = mock_parser

        content = "<robot></robot>"
        engine.load_from_string(content, "urdf")

        mock_parser.AddModelsFromString.assert_called_once_with(content, "urdf")
        assert engine.model_name == "StringLoadedModel"

def test_step(engine):
    # Mock context and diagram
    engine.diagram = MagicMock()
    engine.context = MagicMock()
    engine.context.get_time.return_value = 0.0
    engine.plant.time_step.return_value = 0.001

    with patch("engines.physics_engines.drake.python.drake_physics_engine.analysis.Simulator") as mock_sim_cls:
        mock_sim = MagicMock()
        mock_sim_cls.return_value = mock_sim

        engine.step(0.01)

        mock_sim.Initialize.assert_called_once()
        mock_sim.AdvanceTo.assert_called_once_with(0.01)

def test_get_state(engine):
    engine.plant_context = MagicMock()
    engine.plant.GetPositions.return_value = np.array([1.0, 2.0])
    engine.plant.GetVelocities.return_value = np.array([3.0, 4.0])

    q, v = engine.get_state()

    np.testing.assert_array_equal(q, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(v, np.array([3.0, 4.0]))
    engine.plant.GetPositions.assert_called_once()

def test_compute_mass_matrix(engine):
    engine.plant_context = MagicMock()
    expected_M = np.eye(2)
    engine.plant.CalcMassMatrixViaInverseDynamics.return_value = expected_M

    M = engine.compute_mass_matrix()

    np.testing.assert_array_equal(M, expected_M)
    engine.plant.CalcMassMatrixViaInverseDynamics.assert_called_once()

def test_compute_inverse_dynamics(engine):
    engine.plant_context = MagicMock()
    engine.plant.num_velocities.return_value = 2

    qacc = np.array([1.0, 1.0])
    expected_tau = np.array([10.0, 10.0])
    engine.plant.CalcInverseDynamics.return_value = expected_tau

    tau = engine.compute_inverse_dynamics(qacc)

    np.testing.assert_array_equal(tau, expected_tau)
    engine.plant.CalcInverseDynamics.assert_called_with(
        engine.plant_context,
        qacc,
        engine.plant.MakeMultibodyForces(engine.plant)
    )
