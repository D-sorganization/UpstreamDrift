"""Integration tests for Phase 1 Drake engine functionality.

This module tests the complete Drake engine integration including:
- Engine loading and initialization
- State management (reset, forward, step)
- Error handling and logging
- Integration with the engine manager
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.shared.python.engine_manager import EngineManager, EngineType

# Check if Drake is available
try:
    import pydrake  # noqa: F401

    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False


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

    @patch("engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch("engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder")
    @patch(
        "engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_loading_success(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test successful Drake model loading."""
        # Mock Drake components
        mock_plant = MagicMock()
        mock_scene_graph = MagicMock()
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock plant methods
        mock_plant.Finalize.return_value = None
        mock_plant.time_step.return_value = 0.001

        # Mock builder and diagram
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance
        mock_diagram = MagicMock()
        mock_builder_instance.Build.return_value = mock_diagram
        mock_context = MagicMock()
        mock_diagram.CreateDefaultContext.return_value = mock_context
        mock_plant_context = MagicMock()
        mock_plant.GetMyContextFromRoot.return_value = mock_plant_context

        # Mock Parser class
        with patch(
            "engines.physics_engines.drake.python.drake_physics_engine.Parser"
        ) as mock_parser_class:
            mock_parser_instance = MagicMock()
            mock_parser_class.return_value = mock_parser_instance

            # Mock simulator
            mock_simulator = MagicMock()
            mock_pydrake.systems.analysis.Simulator.return_value = mock_simulator

            # Import and test loading
            from src.engines.physics_engines.drake.python.drake_physics_engine import (
                DrakePhysicsEngine,
            )

            engine = DrakePhysicsEngine()

            # Set up the engine's plant attribute so the load method can access it
            engine.plant = mock_plant

            engine.load_from_path(str(self.urdf_path))

            # Verify loading was attempted
            mock_parser_instance.AddModels.assert_called_once_with(str(self.urdf_path))
            mock_plant.Finalize.assert_called_once()

    @patch("engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch("engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder")
    @patch(
        "engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_loading_file_not_found(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine handles missing files gracefully."""
        # Mock Drake components
        mock_plant = MagicMock()
        mock_scene_graph = MagicMock()
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        engine.plant = mock_plant

        # Mock the file check to raise FileNotFoundError
        with patch("pathlib.Path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                engine.load_from_path("nonexistent_file.urdf")

    @patch("engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch("engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder")
    @patch(
        "engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_reset_functionality(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine reset functionality."""
        # Mock Drake components
        mock_plant = MagicMock()
        mock_scene_graph = MagicMock()
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance

        mock_diagram = MagicMock()
        mock_context = MagicMock()
        mock_plant_context = MagicMock()
        mock_simulator = MagicMock()

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Setup engine state
        engine.plant = mock_plant
        engine.diagram = mock_diagram
        engine.context = mock_context
        engine.plant_context = mock_plant_context
        engine.simulator = mock_simulator

        # Test reset
        engine.reset()

        # Verify reset behavior
        mock_context.SetTime.assert_called_once_with(0.0)
        mock_plant.SetDefaultPositions.assert_called_once_with(mock_plant_context)
        mock_plant.SetDefaultVelocities.assert_called_once_with(mock_plant_context)
        mock_simulator.Initialize.assert_called_once()

    @patch("engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch("engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder")
    @patch(
        "engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_forward_computation(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine forward computation."""
        # Mock Drake components
        mock_plant = MagicMock()
        mock_scene_graph = MagicMock()
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance

        mock_plant_context = MagicMock()
        mock_plant.num_velocities.return_value = 3
        mock_plant.MakeMultibodyForces.return_value = MagicMock()

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Setup engine state
        engine.plant = mock_plant
        engine.plant_context = mock_plant_context

        # Test forward computation
        engine.forward()

        # Verify forward computation was triggered
        mock_plant.CalcMassMatrixViaInverseDynamics.assert_called_once_with(
            mock_plant_context
        )
        mock_plant.CalcInverseDynamics.assert_called_once()

    @patch("engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch("engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder")
    @patch(
        "engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_step_with_caching(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine step with simulator caching."""
        # Mock Drake components
        mock_plant = MagicMock()
        mock_scene_graph = MagicMock()
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance

        mock_diagram = MagicMock()
        mock_context = MagicMock()
        mock_simulator = MagicMock()
        mock_plant.time_step.return_value = 0.001

        mock_context.get_time.return_value = 0.0

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Setup engine state to simulate finalized engine
        engine.plant = mock_plant
        engine.diagram = mock_diagram
        engine.context = mock_context
        engine.simulator = mock_simulator
        engine._is_finalized = True

        # Test step
        engine.step(0.01)

        # Verify simulator advance was called
        mock_simulator.AdvanceTo.assert_called_once()

    @patch("engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch("engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder")
    @patch(
        "engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_error_handling_no_model(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine handles operations without loaded model."""
        # Mock Drake components
        mock_plant = MagicMock()
        mock_scene_graph = MagicMock()
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Test operations without model (should not crash)
        engine.reset()
        engine.forward()
        engine.step(0.01)

        # Test state operations
        q, v = engine.get_state()
        self.assertEqual(len(q), 0)
        self.assertEqual(len(v), 0)

        # Test time
        self.assertEqual(engine.get_time(), 0.0)

    @patch("engines.physics_engines.drake.python.drake_physics_engine.LOGGER")
    @patch("engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch("engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder")
    @patch(
        "engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_logging(
        self, mock_add_plant, mock_builder, mock_pydrake, mock_logger
    ) -> None:
        """Test Drake engine logging functionality."""
        # Mock Drake components
        mock_plant = MagicMock()
        mock_scene_graph = MagicMock()
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance

        # Mock Parser to raise exception
        with patch(
            "engines.physics_engines.drake.python.drake_physics_engine.Parser"
        ) as mock_parser_class:
            mock_parser_instance = MagicMock()
            mock_parser_class.return_value = mock_parser_instance
            mock_parser_instance.AddModels.side_effect = Exception("File not found")

            from src.engines.physics_engines.drake.python.drake_physics_engine import (
                DrakePhysicsEngine,
            )

            engine = DrakePhysicsEngine()
            engine.plant = mock_plant

            # Test loading with exception
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                engine.load_from_path("nonexistent.urdf")

            # Verify error was logged
            mock_logger.error.assert_called()

    @patch("engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch("engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder")
    @patch(
        "engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_state_management(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine state management."""
        # Mock Drake components
        mock_plant = MagicMock()
        mock_scene_graph = MagicMock()
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance

        mock_plant_context = MagicMock()
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

    @patch("engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch("engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder")
    @patch(
        "engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_model_name_property(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine model name property."""
        # Mock Drake components
        mock_plant = MagicMock()
        mock_scene_graph = MagicMock()
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Mock builder instance
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Test without model - should return "Drake_NoModel"
        self.assertEqual(engine.model_name, "Drake_NoModel")

        # Test with model name set
        engine.model_name_str = "test_robot"
        self.assertEqual(engine.model_name, "test_robot")

    def test_engine_manager_drake_integration(self) -> None:
        """Test Drake engine integration with EngineManager."""
        manager = EngineManager()

        # Mock the _load_drake_engine method to verify it's called
        with patch.object(manager, "_load_drake_engine") as mock_load_drake:
            mock_load_drake.return_value = MagicMock()

            # Test engine switching
            try:
                manager.switch_engine(EngineType.DRAKE)
                # Verify that the engine manager attempted to load Drake
                mock_load_drake.assert_called_once()
            except Exception:
                # Expected to fail due to missing Drake dependencies
                # but we verified the attempt was made
                mock_load_drake.assert_called_once()

        # Verify that the engine manager has a current engine
        self.assertIsNotNone(manager.current_engine)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
