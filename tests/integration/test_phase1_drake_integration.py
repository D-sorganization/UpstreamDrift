"""Integration tests for Phase 1 Drake engine functionality.

This module tests the complete Drake engine integration including:
- Engine loading and initialization
- State management (reset, forward, step)
- Error handling and logging
- Integration with the engine manager

Refactored to use shared engine availability module (DRY principle).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.engine_core.engine_availability import DRAKE_AVAILABLE
from src.shared.python.engine_core.engine_manager import EngineManager, EngineType

if DRAKE_AVAILABLE:
    from pydrake.all import DiagramBuilder, Parser
    from pydrake.geometry import SceneGraph
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import Context, Diagram

# MultibodyPlant uses some undocumented Drake APIs (SetDefaultVelocities,
# MakeMultibodyForces) via type-ignore in the production code.  We enumerate
# the attributes the tests rely on so that the mock is constrained yet still
# permits those extra calls.
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
    "num_velocities",
    "MakeMultibodyForces",
]


@unittest.skipUnless(DRAKE_AVAILABLE, "Drake not available")
class TestPhase1DrakeIntegration(unittest.TestCase):
    """Integration tests for Phase 1 Drake engine improvements."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a minimal URDF for testing
        self.test_urdf = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
  </link>
</robot>"""

        self.urdf_path = Path(self.temp_dir) / "test_robot.urdf"
        with open(self.urdf_path, "w") as f:
            f.write(self.test_urdf)

    def test_drake_engine_initialization(self) -> None:
        """Test Drake engine initializes correctly."""
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        self.assertIsNotNone(engine)

    @patch("src.engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder"
    )
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_loading_success(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test successful Drake model loading."""
        # Mock Drake components
        mock_plant = MagicMock(spec=_PLANT_SPEC_ATTRS)
        mock_scene_graph = MagicMock(spec=SceneGraph)
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock plant methods
        mock_plant.Finalize.return_value = None
        mock_plant.time_step.return_value = 0.001

        # Mock builder and diagram
        mock_builder_instance = MagicMock(spec=DiagramBuilder)
        mock_builder.return_value = mock_builder_instance
        mock_diagram = MagicMock(spec=Diagram)
        mock_builder_instance.Build.return_value = mock_diagram
        mock_context = MagicMock(spec=Context)
        mock_diagram.CreateDefaultContext.return_value = mock_context
        mock_plant_context = MagicMock(spec=Context)
        mock_plant.GetMyContextFromRoot.return_value = mock_plant_context

        # Mock Parser class
        with patch(
            "src.engines.physics_engines.drake.python.drake_physics_engine.Parser"
        ) as mock_parser_class:
            mock_parser_instance = MagicMock(spec=Parser)
            mock_parser_class.return_value = mock_parser_instance

            # Mock simulator via the analysis module
            with patch(
                "src.engines.physics_engines.drake.python.drake_physics_engine.analysis"
            ) as mock_analysis:
                mock_simulator = MagicMock(spec=Simulator)
                mock_analysis.Simulator.return_value = mock_simulator

                # Import and test loading
                from src.engines.physics_engines.drake.python.drake_physics_engine import (
                    DrakePhysicsEngine,
                )

                engine = DrakePhysicsEngine()

                # Set up the engine's plant attribute so the load method can access it
                engine.plant = mock_plant

                engine.load_from_path(str(self.urdf_path))

                # Verify loading was attempted (source wraps in Path())
                mock_parser_instance.AddModels.assert_called_once_with(
                    Path(str(self.urdf_path))
                )
                mock_plant.Finalize.assert_called_once()

    @patch("src.engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder"
    )
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_loading_file_not_found(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine handles missing files gracefully."""
        # Mock Drake components
        mock_plant = MagicMock(spec=_PLANT_SPEC_ATTRS)
        mock_scene_graph = MagicMock(spec=SceneGraph)
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock(spec=DiagramBuilder)
        mock_builder.return_value = mock_builder_instance

        # Mock Parser to raise RuntimeError (Drake raises RuntimeError for bad files)
        with patch(
            "src.engines.physics_engines.drake.python.drake_physics_engine.Parser"
        ) as mock_parser_class:
            mock_parser_instance = MagicMock(spec=Parser)
            mock_parser_class.return_value = mock_parser_instance
            mock_parser_instance.AddModels.side_effect = RuntimeError("File not found")

            from src.engines.physics_engines.drake.python.drake_physics_engine import (
                DrakePhysicsEngine,
            )

            engine = DrakePhysicsEngine()
            engine.plant = mock_plant

            with self.assertRaises(RuntimeError):
                engine.load_from_path("nonexistent_file.urdf")

    @patch("src.engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder"
    )
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_reset_functionality(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine reset functionality."""
        # Mock Drake components
        mock_plant = MagicMock(spec=_PLANT_SPEC_ATTRS)
        mock_scene_graph = MagicMock(spec=SceneGraph)
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock(spec=DiagramBuilder)
        mock_builder.return_value = mock_builder_instance

        mock_diagram = MagicMock(spec=Diagram)
        mock_context = MagicMock(spec=Context)
        mock_plant_context = MagicMock(spec=Context)
        mock_simulator = MagicMock(spec=Simulator)

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Setup engine state - must satisfy DBC precondition (is_initialized)
        engine.plant = mock_plant
        engine.diagram = mock_diagram
        engine.context = mock_context
        engine.plant_context = mock_plant_context
        engine.simulator = mock_simulator
        engine._is_finalized = True

        # Test reset
        engine.reset()

        # Verify reset behavior
        mock_context.SetTime.assert_called_once_with(0.0)
        mock_plant.SetDefaultPositions.assert_called_once_with(mock_plant_context)
        mock_plant.SetDefaultVelocities.assert_called_once_with(mock_plant_context)
        mock_simulator.Initialize.assert_called_once()

    @patch("src.engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder"
    )
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_forward_computation(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine forward computation."""
        # Mock Drake components
        mock_plant = MagicMock(spec=_PLANT_SPEC_ATTRS)
        mock_scene_graph = MagicMock(spec=SceneGraph)
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock(spec=DiagramBuilder)
        mock_builder.return_value = mock_builder_instance

        mock_plant_context = MagicMock(spec=Context)
        mock_plant.num_velocities.return_value = 3
        mock_plant.MakeMultibodyForces.return_value = MagicMock(spec=["__call__"])

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Setup engine state - must satisfy DBC precondition (is_initialized)
        engine.plant = mock_plant
        engine.plant_context = mock_plant_context
        engine._is_finalized = True

        # Test forward computation
        engine.forward()

        # Verify forward computation was triggered
        mock_plant.CalcMassMatrixViaInverseDynamics.assert_called_once_with(
            mock_plant_context
        )
        mock_plant.CalcInverseDynamics.assert_called_once()

    @patch("src.engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder"
    )
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_step_with_caching(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine step with simulator caching."""
        # Mock Drake components
        mock_plant = MagicMock(spec=_PLANT_SPEC_ATTRS)
        mock_scene_graph = MagicMock(spec=SceneGraph)
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock(spec=DiagramBuilder)
        mock_builder.return_value = mock_builder_instance

        mock_diagram = MagicMock(spec=Diagram)
        mock_context = MagicMock(spec=Context)
        mock_plant_context = MagicMock(spec=Context)
        mock_simulator = MagicMock(spec=Simulator)
        mock_plant.time_step.return_value = 0.001

        mock_context.get_time.return_value = 0.0

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Setup engine state to simulate finalized engine (DBC precondition)
        engine.plant = mock_plant
        engine.diagram = mock_diagram
        engine.context = mock_context
        engine.plant_context = mock_plant_context
        engine.simulator = mock_simulator
        engine._is_finalized = True

        # Test step
        engine.step(0.01)

        # Verify simulator advance was called
        mock_simulator.AdvanceTo.assert_called_once()

    @patch("src.engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder"
    )
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_error_handling_no_model(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine handles operations without loaded model.

        DBC preconditions on reset/forward/step raise PreconditionError
        when the engine is not initialized (no model loaded).
        """
        # Mock Drake components
        mock_plant = MagicMock(spec=_PLANT_SPEC_ATTRS)
        mock_scene_graph = MagicMock(spec=SceneGraph)
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock(spec=DiagramBuilder)
        mock_builder.return_value = mock_builder_instance

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # DBC preconditions should raise PreconditionError for uninitialized engine
        with self.assertRaises(PreconditionError):
            engine.reset()

        with self.assertRaises(PreconditionError):
            engine.forward()

        with self.assertRaises(PreconditionError):
            engine.step(0.01)

        # get_state has no precondition, returns empty arrays
        q, v = engine.get_state()
        self.assertEqual(len(q), 0)
        self.assertEqual(len(v), 0)

        # Test time
        self.assertEqual(engine.get_time(), 0.0)

    @patch("src.engines.physics_engines.drake.python.drake_physics_engine.logger")
    @patch("src.engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder"
    )
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_logging(
        self, mock_add_plant, mock_builder, mock_pydrake, mock_logger
    ) -> None:
        """Test Drake engine logging functionality."""
        # Mock Drake components
        mock_plant = MagicMock(spec=_PLANT_SPEC_ATTRS)
        mock_scene_graph = MagicMock(spec=SceneGraph)
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock(spec=DiagramBuilder)
        mock_builder.return_value = mock_builder_instance

        # Mock Parser to raise exception
        with patch(
            "src.engines.physics_engines.drake.python.drake_physics_engine.Parser"
        ) as mock_parser_class:
            mock_parser_instance = MagicMock(spec=Parser)
            mock_parser_class.return_value = mock_parser_instance
            mock_parser_instance.AddModels.side_effect = RuntimeError("File not found")

            from src.engines.physics_engines.drake.python.drake_physics_engine import (
                DrakePhysicsEngine,
            )

            engine = DrakePhysicsEngine()
            engine.plant = mock_plant

            # Test loading with exception
            with self.assertRaises(RuntimeError):
                engine.load_from_path("nonexistent.urdf")

            # Verify error was logged
            mock_logger.error.assert_called()

    @patch("src.engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder"
    )
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_state_management(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine state management."""
        # Mock Drake components
        mock_plant = MagicMock(spec=_PLANT_SPEC_ATTRS)
        mock_scene_graph = MagicMock(spec=SceneGraph)
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock(spec=DiagramBuilder)
        mock_builder.return_value = mock_builder_instance

        mock_plant_context = MagicMock(spec=Context)
        mock_plant.GetPositions.return_value = np.array([1.0, 2.0])
        mock_plant.GetVelocities.return_value = np.array([3.0, 4.0])

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Setup engine state
        engine.plant = mock_plant
        engine.plant_context = mock_plant_context

        # Test get_state
        q, v = engine.get_state()

        # Verify state extraction
        self.assertEqual(len(q), 2)
        self.assertEqual(len(v), 2)
        np.testing.assert_array_equal(q, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(v, np.array([3.0, 4.0]))

    @patch("src.engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder"
    )
    @patch(
        "src.engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_model_name_property(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine model name property."""
        # Mock Drake components
        mock_plant = MagicMock(spec=_PLANT_SPEC_ATTRS)
        mock_scene_graph = MagicMock(spec=SceneGraph)
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock(spec=DiagramBuilder)
        mock_builder.return_value = mock_builder_instance

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Test without model - default is empty string
        self.assertEqual(engine.model_name, "")

        # Test with model name set
        engine.model_name_str = "test_robot"
        self.assertEqual(engine.model_name, "test_robot")

    def test_engine_manager_drake_integration(self) -> None:
        """Test Drake engine integration with EngineManager."""
        manager = EngineManager()

        # Mock _load_engine (not _load_drake_engine) and set status to AVAILABLE
        with patch.object(manager, "_load_engine") as mock_load:
            mock_load.return_value = None
            # Must set engine status to AVAILABLE for switch_engine to proceed
            from src.shared.python.engine_core.engine_manager import EngineStatus

            manager.engine_status[EngineType.DRAKE] = EngineStatus.AVAILABLE

            # Test engine switching
            result = manager.switch_engine(EngineType.DRAKE)

            # Verify the manager attempted to load the engine
            mock_load.assert_called_once_with(EngineType.DRAKE)
            self.assertTrue(result)
            self.assertEqual(manager.current_engine, EngineType.DRAKE)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
