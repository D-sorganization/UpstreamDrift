import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock pydrake structure before import
mock_pydrake = MagicMock()
sys.modules["pydrake"] = mock_pydrake
sys.modules["pydrake.systems"] = MagicMock()
sys.modules["pydrake.systems.analysis"] = MagicMock()
sys.modules["pydrake.systems.framework"] = MagicMock()
sys.modules["pydrake.multibody"] = MagicMock()
sys.modules["pydrake.multibody.parsing"] = MagicMock()
sys.modules["pydrake.multibody.plant"] = MagicMock()
sys.modules["pydrake.geometry"] = MagicMock()
sys.modules["pydrake.math"] = MagicMock()
sys.modules["pydrake.all"] = MagicMock()

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

try:
    from engines.physics_engines.drake.python.drake_physics_engine import (
        DrakePhysicsEngine,
    )
except ImportError:
    # Handle case where import logic inside module fails due to complex dependencies
    DrakePhysicsEngine = None


class TestDrakeWrapper(unittest.TestCase):
    def setUp(self):
        if DrakePhysicsEngine is None:
            self.skipTest("DrakePhysicsEngine could not be imported")

        # Patch dependencies used in __init__
        self.patcher1 = patch(
            "engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder"
        )
        self.patcher2 = patch(
            "engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
        )

        self.mock_builder_cls = self.patcher1.start()
        self.mock_add_plant = self.patcher2.start()

        # Setup return values
        self.mock_plant = MagicMock()
        self.mock_scene_graph = MagicMock()
        self.mock_add_plant.return_value = (self.mock_plant, self.mock_scene_graph)

        self.addCleanup(self.patcher1.stop)
        self.addCleanup(self.patcher2.stop)

        # Initialize engine
        self.engine = DrakePhysicsEngine(time_step=0.001)

        # Manually verify internal state setup
        self.engine.diagram = MagicMock()
        self.engine.context = MagicMock()
        self.engine.plant_context = MagicMock()
        # Simulator starts None
        self.engine.simulator = None

    @patch("engines.physics_engines.drake.python.drake_physics_engine.analysis")
    def test_step_caching(self, mock_analysis):
        """Test that Simulator is cached and reused in step()."""
        mock_simulator_class = mock_analysis.Simulator
        mock_simulator_instance = mock_simulator_class.return_value

        # Setup builder returns
        self.engine.builder.Build.return_value = self.engine.diagram
        self.engine.diagram.CreateDefaultContext.return_value = self.engine.context
        self.engine.plant.GetMyContextFromRoot.return_value = self.engine.plant_context

        # 1. Finalize (creates simulator)
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
        self.engine.context.get_time.return_value = 0.0
        self.engine.plant.time_step.return_value = 0.001

        self.engine.step(0.01)

        mock_simulator_class.assert_not_called()
        mock_simulator_instance.Initialize.assert_not_called()
        mock_simulator_instance.AdvanceTo.assert_called()

    def test_reset_logic(self):
        """Test that reset() clears state properly."""
        self.engine.context = MagicMock()
        self.engine.plant_context = MagicMock()
        self.engine.plant = MagicMock()
        self.engine.simulator = MagicMock()

        self.engine.reset()

        # Verify context time reset
        self.engine.context.SetTime.assert_called_with(0.0)

        # Verify default context usage (from my fix)
        self.engine.plant.SetDefaultContext.assert_called_with(
            self.engine.plant_context
        )

        # Verify simulator initialization
        self.engine.simulator.Initialize.assert_called_once()

    def test_forward_lazy_eval(self):
        """Test forward implementation (Lazy Evaluation Pattern)."""
        # Ensure forward() is a no-op that relies on Drake's internal lazy evaluation mechanisms
        # It should NOT eagerly compute anything or invalidate the context manually.
        self.engine.context = MagicMock()
        self.engine.forward()

        # Assert no aggressive computation was triggered
        self.engine.context.SetTime.assert_not_called()
        self.engine.context.SetContinuousState.assert_not_called()


if __name__ == "__main__":
    unittest.main()
