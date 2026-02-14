import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.engine_core.engine_availability import (
    DRAKE_AVAILABLE,
    skip_if_unavailable,
)

# Skip entire module if Drake is not installed - mocking pydrake at module level
# is unreliable and leads to AttributeError on patched module globals
pytestmark = skip_if_unavailable("drake")

# Drake engine module paths (both with and without src. prefix) that need cleanup
_DRAKE_ENGINE_MODULES = [
    "engines",
    "engines.physics_engines",
    "engines.physics_engines.drake",
    "engines.physics_engines.drake.python",
    "engines.physics_engines.drake.python.drake_physics_engine",
    "src.engines.physics_engines.drake",
    "src.engines.physics_engines.drake.python",
    "src.engines.physics_engines.drake.python.drake_physics_engine",
]

if DRAKE_AVAILABLE:
    from pydrake.all import DiagramBuilder, Parser
    from pydrake.geometry import SceneGraph
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import Context

# MultibodyPlant uses undocumented Drake APIs; enumerate tested attributes.
_PLANT_SPEC_ATTRS = [
    "Finalize",
    "time_step",
    "GetMyContextFromRoot",
    "SetDefaultPositions",
    "SetDefaultVelocities",
    "GetPositions",
    "GetVelocities",
    "CalcMassMatrixViaInverseDynamics",
    "CalcInverseDynamics",
    "CalcGravityGeneralizedForces",
    "num_velocities",
    "MakeMultibodyForces",
]


# Mock classes that need to be defined before importing the engine
class MockPhysicsEngine:
    pass


def _cleanup_drake_modules():
    """Clean up drake engine modules to prevent pollution of other tests."""
    for module_name in _DRAKE_ENGINE_MODULES:
        sys.modules.pop(module_name, None)


@pytest.fixture(autouse=True, scope="module")
def mock_drake_dependencies():
    """Fixture to mock pydrake and interfaces safely for the duration of this module.

    Also cleanup drake engine modules after all tests to prevent pollution of
    test_drake_wrapper.py. When drake_physics_engine is imported, it pollutes
    sys.modules with parent packages. Then test_drake_wrapper.py fails when trying
    to patch src.engines.physics_engines.drake.python.drake_physics_engine because
    the parent package exists but drake_physics_engine was imported earlier.
    """
    mock_pydrake = MagicMock()
    mock_interfaces = MagicMock()
    mock_interfaces.PhysicsEngine = MockPhysicsEngine

    with patch.dict(
        "sys.modules",
        {
            "pydrake": mock_pydrake,
            "pydrake.math": MagicMock(),
            "pydrake.multibody": MagicMock(),
            "pydrake.multibody.plant": MagicMock(),
            "pydrake.multibody.parsing": MagicMock(),
            "pydrake.systems": MagicMock(),
            "pydrake.systems.framework": MagicMock(),
            "pydrake.systems.analysis": MagicMock(),
            "pydrake.all": MagicMock(),
            "shared.python.interfaces": mock_interfaces,
        },
    ):
        yield mock_pydrake, mock_interfaces

    # Clean up drake engine modules to prevent pollution
    _cleanup_drake_modules()


# Also cleanup at module teardown in case the fixture doesn't run for all test scenarios
def teardown_module():
    """Module-level teardown to ensure cleanup happens."""
    _cleanup_drake_modules()


@pytest.fixture(scope="module")
def DrakePhysicsEngineClass(mock_drake_dependencies):
    """Fixture to provide the DrakePhysicsEngine class with mocked dependencies."""
    # Ensure module is imported
    import engines.physics_engines.drake.python.drake_physics_engine as mod

    # Manually patch the module's globals to use our mocks
    # This avoids reload() which corrupts sys.modules state
    mock_pydrake, mock_interfaces = mock_drake_dependencies

    # Save originals
    original_pydrake = getattr(mod, "pydrake", None)
    original_interfaces = getattr(mod, "interfaces", None)

    # Inject mocks
    mod.pydrake = mock_pydrake  # type: ignore[attr-defined]
    mod.interfaces = mock_interfaces  # type: ignore[attr-defined]

    yield mod.DrakePhysicsEngine

    # Restore (optional but good practice)
    if original_pydrake:
        mod.pydrake = original_pydrake  # type: ignore[attr-defined]
    if original_interfaces:
        mod.interfaces = original_interfaces  # type: ignore[attr-defined]


@pytest.fixture
def engine(DrakePhysicsEngineClass):
    """Fixture to provide an uninitialized DrakePhysicsEngine instance."""
    with patch(
        "engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    ) as mock_add:
        mock_plant = MagicMock(spec=_PLANT_SPEC_ATTRS)
        mock_scene_graph = MagicMock(spec=SceneGraph)
        mock_add.return_value = (mock_plant, mock_scene_graph)

        eng = DrakePhysicsEngineClass()
        eng.plant = mock_plant
        eng.scene_graph = mock_scene_graph
        eng.builder = MagicMock(spec=DiagramBuilder)
        return eng


@pytest.fixture
def initialized_engine(engine):
    """Fixture providing a DrakePhysicsEngine that satisfies DBC preconditions.

    Sets _is_finalized=True and plant_context so @precondition(is_initialized)
    passes for step/reset/forward/compute_* methods.
    """
    engine._is_finalized = True
    engine.plant_context = MagicMock(spec=Context)
    engine.context = MagicMock(spec=Context)
    engine.simulator = MagicMock(spec=Simulator)
    return engine


def test_initialization(engine):
    assert engine.plant is not None
    assert engine.builder is not None
    assert not engine._is_finalized


def test_load_from_path(engine):
    from pathlib import Path as StdPath

    with patch(
        "engines.physics_engines.drake.python.drake_physics_engine.Parser"
    ) as mock_parser_cls:
        mock_parser = MagicMock(spec=Parser)
        mock_parser_cls.return_value = mock_parser

        path = "test_model.urdf"
        engine.load_from_path(path)

        # Source wraps the path in Path() before passing to AddModels
        mock_parser.AddModels.assert_called_once_with(StdPath(path))
        assert engine.model_name == "test_model"
        # Should ensure finalized
        engine.plant.Finalize.assert_called_once()
        engine.builder.Build.assert_called_once()


def test_load_from_string(engine):
    with patch(
        "engines.physics_engines.drake.python.drake_physics_engine.Parser"
    ) as mock_parser_cls:
        mock_parser = MagicMock(spec=Parser)
        mock_parser_cls.return_value = mock_parser

        content = "<robot></robot>"
        engine.load_from_string(content, "urdf")

        mock_parser.AddModelsFromString.assert_called_once_with(content, "urdf")
        assert engine.model_name == "StringLoadedModel"


def test_step(initialized_engine):
    """Test step method on an initialized engine (DBC: requires is_initialized)."""
    engine = initialized_engine

    engine.context.get_time.return_value = 0.0
    engine.plant.time_step.return_value = 0.001

    engine.step(0.01)

    engine.simulator.AdvanceTo.assert_called_once_with(0.01)


def test_get_state(initialized_engine):
    """Test get_state on an initialized engine."""
    engine = initialized_engine

    engine.plant.GetPositions.return_value = np.array([1.0, 2.0])
    engine.plant.GetVelocities.return_value = np.array([3.0, 4.0])

    q, v = engine.get_state()

    np.testing.assert_array_equal(q, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(v, np.array([3.0, 4.0]))
    engine.plant.GetPositions.assert_called_once()


def test_compute_mass_matrix(initialized_engine):
    """Test compute_mass_matrix on an initialized engine (DBC: requires is_initialized)."""
    engine = initialized_engine

    expected_M = np.eye(2)
    engine.plant.CalcMassMatrixViaInverseDynamics.return_value = expected_M

    M = engine.compute_mass_matrix()

    np.testing.assert_array_equal(M, expected_M)
    engine.plant.CalcMassMatrixViaInverseDynamics.assert_called_once()


def test_compute_inverse_dynamics(initialized_engine):
    """Test compute_inverse_dynamics on an initialized engine (DBC: requires is_initialized)."""
    engine = initialized_engine

    engine.plant.num_velocities.return_value = 2

    qacc = np.array([1.0, 1.0])
    expected_tau = np.array([10.0, 10.0])
    engine.plant.CalcInverseDynamics.return_value = expected_tau

    tau = engine.compute_inverse_dynamics(qacc)

    np.testing.assert_array_equal(tau, expected_tau)
    engine.plant.CalcInverseDynamics.assert_called_with(
        engine.plant_context, qacc, engine.plant.MakeMultibodyForces(engine.plant)
    )
