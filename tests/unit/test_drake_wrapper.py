import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.engine_core.engine_availability import DRAKE_AVAILABLE

if DRAKE_AVAILABLE:
    from pydrake.geometry import SceneGraph
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import Context, Diagram

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

# Mock pydrake using patch.dict (auto-cleans) to allow DrakePhysicsEngine import.
_PYDRAKE_KEYS = [
    "pydrake",
    "pydrake.systems",
    "pydrake.systems.analysis",
    "pydrake.systems.framework",
    "pydrake.multibody",
    "pydrake.multibody.parsing",
    "pydrake.multibody.plant",
    "pydrake.geometry",
    "pydrake.math",
    "pydrake.all",
]

_ENGINE_MOD_NAME = "src.engines.physics_engines.drake.python.drake_physics_engine"

_pydrake_mocks = {k: MagicMock() for k in _PYDRAKE_KEYS}
_drake_engine_module = None  # will hold module reference for @patch usage

with patch.dict(sys.modules, _pydrake_mocks):
    with patch(
        "src.shared.python.engine_core.engine_availability.DRAKE_AVAILABLE", True
    ):
        # Force re-import of drake_physics_engine with DRAKE_AVAILABLE=True
        sys.modules.pop(_ENGINE_MOD_NAME, None)

        try:
            from src.engines.physics_engines.drake.python import (
                drake_physics_engine as _drake_engine_module,
            )

            DrakePhysicsEngine = _drake_engine_module.DrakePhysicsEngine
        except ImportError:
            DrakePhysicsEngine = None  # type: ignore[assignment,misc]

    # Remove the mock-backed engine module to prevent pollution during
    # integration tests that run before this file's tests.
    sys.modules.pop(_ENGINE_MOD_NAME, None)


@pytest.fixture(autouse=True)
def _isolate_drake_module_state():
    """Prevent module-level Drake mocks from leaking across tests."""
    engine_before = sys.modules.get(_ENGINE_MOD_NAME)
    pydrake_before = {key: sys.modules.get(key) for key in _PYDRAKE_KEYS}
    yield

    if engine_before is None:
        sys.modules.pop(_ENGINE_MOD_NAME, None)
    else:
        sys.modules[_ENGINE_MOD_NAME] = engine_before

    for key, value in pydrake_before.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value


class TestDrakeWrapper(unittest.TestCase):
    def setUp(self):
        if DrakePhysicsEngine is None:
            self.skipTest("DrakePhysicsEngine could not be imported")

        # Temporarily restore the engine module so @patch decorators can find it
        if _drake_engine_module is not None:
            sys.modules[_ENGINE_MOD_NAME] = _drake_engine_module
            self.addCleanup(lambda: sys.modules.pop(_ENGINE_MOD_NAME, None))

        # Patch dependencies used in __init__
        self.patcher1 = patch(
            "src.engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder"
        )
        self.patcher2 = patch(
            "src.engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
        )

        self.mock_builder_cls = self.patcher1.start()
        self.mock_add_plant = self.patcher2.start()

        # Setup return values
        self.mock_plant = MagicMock(spec=_PLANT_SPEC_ATTRS)
        self.mock_scene_graph = (
            MagicMock(spec=SceneGraph) if DRAKE_AVAILABLE else MagicMock()
        )
        self.mock_add_plant.return_value = (self.mock_plant, self.mock_scene_graph)

        self.addCleanup(self.patcher1.stop)
        self.addCleanup(self.patcher2.stop)

        # Initialize engine
        self.engine = DrakePhysicsEngine(time_step=0.001)

        # Manually verify internal state setup
        self.engine.diagram = (
            MagicMock(spec=Diagram) if DRAKE_AVAILABLE else MagicMock()
        )
        self.engine.context = (
            MagicMock(spec=Context) if DRAKE_AVAILABLE else MagicMock()
        )
        self.engine.plant_context = (
            MagicMock(spec=Context) if DRAKE_AVAILABLE else MagicMock()
        )
        self.engine._is_finalized = True
        # Simulator starts None
        self.engine.simulator = None

    def test_step_caching(self):
        """Test that Simulator is cached and reused in step()."""
        # Patch analysis directly on the module object to avoid sys.modules
        # lookup issues that occur in full-suite ordering.
        mock_analysis = MagicMock(spec=["Simulator"])
        original_analysis = getattr(_drake_engine_module, "analysis", None)
        _drake_engine_module.analysis = mock_analysis
        self.addCleanup(setattr, _drake_engine_module, "analysis", original_analysis)

        mock_simulator_class = mock_analysis.Simulator
        mock_simulator_instance = mock_simulator_class.return_value

        # Setup builder returns
        self.engine.builder.Build.return_value = self.engine.diagram
        if self.engine.diagram is not None:
            self.engine.diagram.CreateDefaultContext.return_value = self.engine.context
        self.engine.plant.GetMyContextFromRoot.return_value = self.engine.plant_context

        # 1. Finalize (creates simulator) - reset flag so finalization actually runs
        self.engine._is_finalized = False
        self.engine._ensure_finalized()

        mock_simulator_class.assert_called_once_with(
            self.engine.diagram, self.engine.context
        )
        mock_simulator_instance.Initialize.assert_called_once()
        self.assertIsNotNone(self.engine.simulator)

        # Reset mock counts
        mock_simulator_class.reset_mock()
        mock_simulator_instance.Initialize.reset_mock()

        # 2. Step (should reuse simulator)
        if self.engine.context is not None:
            self.engine.context.get_time.return_value = 0.0
        self.engine.plant.time_step.return_value = 0.001

        self.engine.step(0.01)

        mock_simulator_class.assert_not_called()
        mock_simulator_instance.Initialize.assert_not_called()
        mock_simulator_instance.AdvanceTo.assert_called()

    def test_reset_logic(self):
        """Test that reset() properly resets state to defaults."""
        self.engine.context = (
            MagicMock(spec=Context) if DRAKE_AVAILABLE else MagicMock()
        )
        self.engine.plant_context = (
            MagicMock(spec=Context) if DRAKE_AVAILABLE else MagicMock()
        )
        self.engine.plant = MagicMock(spec=_PLANT_SPEC_ATTRS)
        self.engine.simulator = (
            MagicMock(spec=Simulator) if DRAKE_AVAILABLE else MagicMock()
        )

        self.engine.reset()

        # Verify context time reset
        self.engine.context.SetTime.assert_called_with(0.0)

        # Verify default positions and velocities are set (from my fix)
        self.engine.plant.SetDefaultPositions.assert_called_with(
            self.engine.plant_context
        )
        self.engine.plant.SetDefaultVelocities.assert_called_with(
            self.engine.plant_context
        )

        # Verify simulator re-initialization
        self.engine.simulator.Initialize.assert_called_once()

    def test_forward_computation(self):
        """Test forward() triggers computation of derived quantities."""
        self.engine.plant_context = (
            MagicMock(spec=Context) if DRAKE_AVAILABLE else MagicMock()
        )
        self.engine.plant = MagicMock(spec=_PLANT_SPEC_ATTRS)

        # Setup plant methods
        self.engine.plant.num_velocities.return_value = 3
        self.engine.plant.CalcMassMatrixViaInverseDynamics.return_value = MagicMock(
            spec=["shape"]
        )
        self.engine.plant.CalcInverseDynamics.return_value = MagicMock(spec=["shape"])
        self.engine.plant.MakeMultibodyForces.return_value = MagicMock(
            spec=["mutable_generalized_forces"]
        )

        self.engine.forward()

        # Verify mass matrix computation was triggered (forces forward dynamics)
        self.engine.plant.CalcMassMatrixViaInverseDynamics.assert_called_once_with(
            self.engine.plant_context
        )

        # Verify bias forces computation was triggered (ensures kinematics updated)
        self.engine.plant.CalcInverseDynamics.assert_called_once()

    def test_forward_with_no_context(self):
        """Test forward() raises PreconditionError when context is missing."""
        from src.shared.python.core.contracts import PreconditionError

        self.engine.plant_context = None

        # With Design by Contract, forward() requires is_initialized,
        # which checks plant_context is not None
        with self.assertRaises(PreconditionError):
            self.engine.forward()


if __name__ == "__main__":
    unittest.main()
