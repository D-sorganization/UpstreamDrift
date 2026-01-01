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

from shared.python.engine_manager import EngineManager


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

    @patch("engines.physics_engines.drake.python.drake_physics_engine.pydrake")
    @patch("engines.physics_engines.drake.python.drake_physics_engine.DiagramBuilder")
    @patch(
        "engines.physics_engines.drake.python.drake_physics_engine.AddMultibodyPlantSceneGraph"
    )
    def test_drake_engine_initialization(
        self, mock_add_plant, mock_builder, mock_pydrake
    ) -> None:
        """Test Drake engine initializes correctly."""
        # Mock Drake components
        mock_plant = MagicMock()
        mock_scene_graph = MagicMock()
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        # Import and create engine
        from engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        self.assertIsNotNone(engine)
        mock_builder.assert_called_once()
        mock_add_plant.assert_called_once()

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

        mock_parser = MagicMock()
        mock_plant.get_parser.return_value = mock_parser
        mock_plant.Finalize.return_value = None

        mock_diagram = MagicMock()
        mock_context = MagicMock()
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance
        mock_builder_instance.Build.return_value = mock_diagram
        mock_diagram.CreateDefaultContext.return_value = mock_context

        # Import and test loading
        from engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        engine.load_from_path(str(self.urdf_path))

        # Verify loading was attempted
        mock_parser.AddModels.assert_called_once()
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

        from engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

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

        mock_diagram = MagicMock()
        mock_context = MagicMock()
        mock_default_context = MagicMock()

        from engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Setup engine state
        engine.diagram = mock_diagram
        engine.context = mock_context
        mock_diagram.CreateDefaultContext.return_value = mock_default_context
        mock_default_context.Clone.return_value = mock_context

        # Test reset
        engine.reset()

        # Verify reset behavior
        mock_diagram.CreateDefaultContext.assert_called_once()

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

        mock_diagram = MagicMock()
        mock_context = MagicMock()

        from engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Setup engine state
        engine.diagram = mock_diagram
        engine.context = mock_context

        # Test forward computation
        engine.forward()

        # Verify forward computation was triggered
        mock_diagram.ForcedPublish.assert_called_once_with(mock_context)

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

        mock_diagram = MagicMock()
        mock_context = MagicMock()
        mock_simulator = MagicMock()

        mock_pydrake.systems.analysis.Simulator.return_value = mock_simulator
        mock_context.get_time.return_value = 0.0
        mock_context.SetTime = MagicMock()

        from engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Setup engine state
        engine.diagram = mock_diagram
        engine.context = mock_context

        # Test first step (should create simulator)
        engine.step(0.01)

        # Verify simulator was created and cached
        mock_pydrake.systems.analysis.Simulator.assert_called_once()
        self.assertIsNotNone(engine._cached_simulator)

        # Test second step (should reuse simulator)
        mock_pydrake.systems.analysis.Simulator.reset_mock()
        engine.step(0.01)

        # Verify simulator was not recreated
        mock_pydrake.systems.analysis.Simulator.assert_not_called()

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

        from engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Test operations without model
        engine.reset()  # Should not crash
        engine.forward()  # Should not crash
        engine.step(0.01)  # Should not crash

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

        from engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Test loading with missing file
        try:
            engine.load_from_path("nonexistent.urdf")
        except FileNotFoundError:
            pass

        # Verify error was logged
        mock_logger.error.assert_called()

    def test_engine_manager_drake_integration(self) -> None:
        """Test Drake engine integration with EngineManager."""
        manager = EngineManager()

        # Test engine switching
        with patch.object(manager, "_load_drake_engine") as mock_load:
            mock_load.return_value = True
            manager.switch_engine("drake")

            # Verify engine loading was attempted
            mock_load.assert_called_once()

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

        mock_diagram = MagicMock()
        mock_context = MagicMock()
        mock_plant.num_positions.return_value = 2
        mock_plant.num_velocities.return_value = 2

        # Mock state vectors
        mock_context.get_continuous_state_vector.return_value.CopyToVector.return_value = np.array(
            [1, 2, 3, 4]
        )

        from engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Setup engine state
        engine.diagram = mock_diagram
        engine.context = mock_context

        # Test get_state
        q, v = engine.get_state()

        # Verify state extraction
        self.assertEqual(len(q), 2)
        self.assertEqual(len(v), 2)

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

        from engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()

        # Test without model
        self.assertEqual(engine.model_name, "Drake_NoModel")

        # Test with mock model
        mock_diagram = MagicMock()
        mock_diagram.get_name.return_value = "test_robot"
        engine.diagram = mock_diagram

        self.assertEqual(engine.model_name, "test_robot")

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
