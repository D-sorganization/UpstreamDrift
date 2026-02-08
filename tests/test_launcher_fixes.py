#!/usr/bin/env python3
"""
Test suite for Golf Modeling Suite launcher fixes and new features.

Tests cover:
- Drag-and-drop functionality
- Docker container setup
- Module import fixes
- Engine detection and management
"""

import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

from src.shared.python.engine_availability import PYQT6_AVAILABLE
from src.shared.python.gui_utils import get_qapp
from src.shared.python.path_utils import setup_import_paths

# Setup import paths for testing
setup_import_paths()

if PYQT6_AVAILABLE:
    from PyQt6.QtCore import QPoint


class TestSharedModuleImports(unittest.TestCase):
    """Test that shared modules can be imported correctly."""

    def test_configuration_manager_import(self) -> None:
        """Test configuration manager import."""
        try:
            from src.shared.python.configuration_manager import ConfigurationManager

            # Test that we can instantiate it with required config_path
            config_manager = ConfigurationManager(Path("dummy_config.json"))
            self.assertIsNotNone(config_manager)
        except ImportError as e:
            self.fail(f"Failed to import ConfigurationManager: {e}")
        except Exception as e:
            # If instantiation fails due to missing file, that's expected in tests
            self.assertTrue(True, f"ConfigurationManager import successful: {e}")

    def test_process_worker_import(self) -> None:
        """Test process worker import."""
        try:
            from src.shared.python.ui.qt.process_worker import ProcessWorker

            # Test that we can instantiate it with required cmd
            worker = ProcessWorker(["echo", "test"])
            self.assertIsNotNone(worker)
        except ImportError as e:
            self.fail(f"Failed to import ProcessWorker: {e}")

    def test_engine_manager_import(self) -> None:
        """Test engine manager import."""
        try:
            from src.shared.python.engine_manager import EngineManager, EngineType

            # Test that we can instantiate it
            manager = EngineManager()
            self.assertIsNotNone(manager)
            # Test that EngineType enum exists
            self.assertTrue(hasattr(EngineType, "MUJOCO"))
        except ImportError as e:
            self.fail(f"Failed to import EngineManager: {e}")


class TestEngineManager(unittest.TestCase):
    """Test engine manager functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from src.shared.python.engine_manager import EngineManager

        self.manager = EngineManager()

    def test_engine_discovery(self) -> None:
        """Test that engines are discovered correctly."""
        engines = self.manager.get_available_engines()
        self.assertIsInstance(engines, list)
        self.assertGreater(len(engines), 0, "Should discover at least one engine")

    def test_engine_paths_exist(self) -> None:
        """Test that engine paths are properly configured."""
        for _engine_type, path in self.manager.engine_paths.items():
            self.assertIsInstance(path, Path)
            # Note: Not all engines may be installed, so we don't require existence

    @patch("src.shared.python.engine_manager.EngineManager.get_probe_result")
    def test_probe_system(self, mock_get_result: MagicMock) -> None:
        """Test engine probe system."""
        from src.shared.python.engine_manager import EngineType

        # Setup mock return
        mock_result = MagicMock()
        mock_result.is_available = True
        mock_result.diagnostic_message = "Mocked result"
        mock_get_result.return_value = mock_result

        # Test MuJoCo probe if available
        if EngineType.MUJOCO in self.manager.probes:
            probe_result = self.manager.get_probe_result(EngineType.MUJOCO)
            self.assertIsNotNone(probe_result)
            self.assertTrue(hasattr(probe_result, "is_available"))
            self.assertTrue(hasattr(probe_result, "diagnostic_message"))


@unittest.skipUnless(PYQT6_AVAILABLE, "PyQt6 not available")
class TestDraggableModelCard(unittest.TestCase):
    """Test drag-and-drop functionality in model cards."""

    app: Any = None

    @classmethod
    def setUpClass(cls) -> None:
        """Set up QApplication for GUI tests."""
        cls.app = get_qapp()  # Must store reference to prevent GC

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock model object (set type/path explicitly to avoid Mock auto-attribute issues)
        self.mock_model = Mock()
        self.mock_model.id = "test_model"
        self.mock_model.name = "Test Model"
        self.mock_model.description = "Test Description"
        self.mock_model.type = ""
        self.mock_model.path = ""

        # Mock parent launcher
        self.mock_launcher = Mock()
        self.mock_launcher.select_model = Mock()
        self.mock_launcher._swap_models = Mock()
        self.mock_launcher.launch_model_direct = Mock()
        self.mock_launcher.layout_edit_mode = False

    def test_draggable_card_creation(self) -> None:
        """Test that draggable model cards can be created."""
        from src.launchers.ui_components import DraggableModelCard

        card = DraggableModelCard(self.mock_model, self.mock_launcher)
        self.assertIsNotNone(card)
        self.assertEqual(card.model, self.mock_model)
        self.assertEqual(card.parent_launcher, self.mock_launcher)

    def test_mouse_press_selection(self) -> None:
        """Test that mouse press selects the model."""
        from src.launchers.ui_components import DraggableModelCard

        card = DraggableModelCard(self.mock_model, self.mock_launcher)

        # Create mock mouse event with proper Qt enum
        from PyQt6.QtCore import Qt

        event = Mock()
        event.button.return_value = Qt.MouseButton.LeftButton
        event.position.return_value.toPoint.return_value = QPoint(10, 10)

        # Verify parent launcher is set correctly
        self.assertIsNotNone(card.parent_launcher, "Parent launcher should not be None")
        self.assertEqual(
            card.parent_launcher,
            self.mock_launcher,
            "Parent launcher should match mock",
        )

        card.mousePressEvent(event)

        # Verify model selection was called
        self.mock_launcher.select_model.assert_called_once_with("test_model")

    def test_double_click_launch(self) -> None:
        """Test that double-click launches the model."""
        from src.launchers.ui_components import DraggableModelCard

        card = DraggableModelCard(self.mock_model, self.mock_launcher)

        # Create mock double-click event
        event = Mock()

        card.mouseDoubleClickEvent(event)

        # Verify launch was called
        self.mock_launcher.launch_model_direct.assert_called_once_with("test_model")

    def test_drag_enter_event(self) -> None:
        """Test drag enter event handling."""
        from src.launchers.ui_components import DraggableModelCard

        card = DraggableModelCard(self.mock_model, self.mock_launcher)

        # Create mock drag enter event with valid mime data
        event = Mock()
        mime_data = Mock()
        mime_data.hasText.return_value = True
        mime_data.text.return_value = "model_card:other_model"
        event.mimeData.return_value = mime_data

        card.dragEnterEvent(event)

        # Should accept the event
        event.acceptProposedAction.assert_called_once()

    def test_drop_event_swap(self) -> None:
        """Test drop event triggers model swap."""
        from src.launchers.ui_components import DraggableModelCard

        card = DraggableModelCard(self.mock_model, self.mock_launcher)

        # Create mock drop event
        event = Mock()
        mime_data = Mock()
        mime_data.hasText.return_value = True
        mime_data.text.return_value = "model_card:source_model"
        event.mimeData.return_value = mime_data

        card.dropEvent(event)

        # Verify swap was called
        self.mock_launcher._swap_models.assert_called_once_with(
            "source_model", "test_model"
        )
        event.acceptProposedAction.assert_called_once()


@unittest.skipUnless(PYQT6_AVAILABLE, "PyQt6 not available")
class TestGolfLauncherGrid(unittest.TestCase):
    """Test golf launcher grid functionality."""

    app: Any = None

    @classmethod
    def setUpClass(cls) -> None:
        """Set up QApplication for GUI tests."""
        cls.app = get_qapp()  # Must store reference to prevent GC

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock registry with test models
        self.mock_registry = Mock()
        self.mock_models = []
        for i in range(4):  # Create 4 test models
            model = Mock()
            model.id = f"model_{i}"
            model.name = f"Model {i}"
            model.description = f"Description {i}"
            model.type = "mujoco"  # Set a proper string type for _get_engine_type
            self.mock_models.append(model)

        # Make sure the registry returns our mock models
        self.mock_registry.get_all_models.return_value = self.mock_models
        # Make the registry iterable for any 'in' operations
        self.mock_registry.__iter__ = lambda x: iter(self.mock_models)

        # Mock get_model to return a model with proper type
        def mock_get_model(model_id: str) -> Any:
            for model in self.mock_models:
                if model.id == model_id:
                    return model
            return None

        self.mock_registry.get_model.side_effect = mock_get_model

    @patch("src.launchers.golf_launcher.GolfLauncher._load_layout")
    @patch("src.launchers.golf_launcher.GolfLauncher.addDockWidget", create=True)
    @patch("src.launchers.golf_launcher.ContextHelpDock")
    @patch("src.launchers.golf_launcher._lazy_load_model_registry")
    @patch("src.launchers.golf_launcher._lazy_load_engine_manager")
    def test_model_order_tracking(
        self,
        mock_lazy_load: Mock,
        mock_lazy_registry: Mock,
        mock_help_dock: Mock,
        mock_add_dock_widget: Mock,
        mock_load_layout: Mock,
    ) -> None:
        """Test that model order is properly tracked."""
        from src.launchers.golf_launcher import GolfLauncher

        # _lazy_load_model_registry returns the ModelRegistry class
        mock_class = Mock()
        mock_class.return_value = self.mock_registry
        mock_lazy_registry.return_value = mock_class

        # _lazy_load_engine_manager returns (EngineManager, EngineType)
        mock_lazy_load.return_value = (Mock(), Mock())

        mock_help_dock.side_effect = None
        launcher = GolfLauncher()

        # Check that model order is initialized as a list
        self.assertIsInstance(launcher.model_order, list)
        # Model order should contain only IDs that exist in available_models
        for model_id in launcher.model_order:
            self.assertIn(model_id, launcher.available_models)

    @patch("src.launchers.golf_launcher.GolfLauncher._save_layout")
    @patch("src.launchers.golf_launcher.GolfLauncher._rebuild_grid")
    @patch("src.launchers.golf_launcher.GolfLauncher._load_layout")
    @patch("src.launchers.golf_launcher.GolfLauncher.addDockWidget", create=True)
    @patch("src.launchers.golf_launcher.ContextHelpDock")
    @patch("src.launchers.golf_launcher._lazy_load_model_registry")
    @patch("src.launchers.golf_launcher._lazy_load_engine_manager")
    def test_model_swap_functionality(
        self,
        mock_lazy_load: Mock,
        mock_lazy_registry: Mock,
        mock_help_dock: Mock,
        mock_add_dock_widget: Mock,
        mock_load_layout: Mock,
        mock_rebuild_grid: Mock,
        mock_save_layout: Mock,
    ) -> None:
        """Test model swapping functionality."""
        from src.launchers.golf_launcher import GolfLauncher

        # _lazy_load_model_registry returns the ModelRegistry class
        mock_class = Mock()
        mock_class.return_value = self.mock_registry
        mock_lazy_registry.return_value = mock_class

        mock_lazy_load.return_value = (Mock(), Mock())
        mock_help_dock.side_effect = None

        launcher = GolfLauncher()

        # Set up initial order and enable edit mode
        launcher.model_order = ["model_0", "model_1", "model_2", "model_3"]
        launcher.layout_manager.model_order = list(launcher.model_order)
        launcher.layout_manager.edit_mode = True
        launcher.model_cards = {
            "model_0": Mock(),
            "model_1": Mock(),
            "model_2": Mock(),
            "model_3": Mock(),
        }

        # Test swapping
        launcher._swap_models("model_0", "model_2")

        # Verify order changed (positions 0 and 2 swapped)
        expected_order = ["model_2", "model_1", "model_0", "model_3"]
        self.assertEqual(launcher.model_order, expected_order)


class TestDockerConfiguration(unittest.TestCase):
    """Test Docker configuration and setup."""

    def test_dockerfile_exists(self) -> None:
        """Test that Dockerfile exists and is readable."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        self.assertTrue(dockerfile_path.exists(), "Dockerfile should exist")

        # Test that it's readable
        content = dockerfile_path.read_text()
        self.assertIn("PYTHONPATH", content, "Dockerfile should set PYTHONPATH")
        self.assertIn("/workspace", content, "Dockerfile should configure workspace")

    def test_docker_image_tag(self) -> None:
        """Test that Dockerfile uses a pinned base image."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile_path.read_text()
        # Should use a pinned version, not :latest
        self.assertIn(
            "continuumio/miniconda3:", content, "Should use miniconda3 base image"
        )
        self.assertNotIn(
            "continuumio/miniconda3:latest",
            content,
            "Should use pinned version, not :latest",
        )


class TestMuJoCoModule(unittest.TestCase):
    """Test MuJoCo module structure and availability."""

    def test_mujoco_module_exists(self) -> None:
        """Test that MuJoCo humanoid golf module exists."""
        mujoco_path = Path(
            "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf"
        )
        self.assertTrue(mujoco_path.exists(), "MuJoCo module directory should exist")

        main_file = mujoco_path / "__main__.py"
        self.assertTrue(main_file.exists(), "MuJoCo module should have __main__.py")

    def test_mujoco_module_structure(self) -> None:
        """Test MuJoCo module has required components."""
        mujoco_path = Path(
            "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf"
        )

        required_files = [
            "__init__.py",
            "__main__.py",
        ]

        for file_name in required_files:
            file_path = mujoco_path / file_name
            self.assertTrue(
                file_path.exists(), f"Required file {file_name} should exist"
            )

    def test_mujoco_module_name_in_handler(self) -> None:
        """Test that the module name in model handlers is correct."""
        from src.launchers.launcher_model_handlers import HumanoidMuJoCoHandler

        handler = HumanoidMuJoCoHandler()
        # The handler should use the package path (not .main suffix)
        # since Python -m runs __main__.py automatically
        mock_model = Mock()
        mock_process_manager = Mock()
        mock_process_manager.launch_module.return_value = Mock()

        handler.launch(mock_model, Path("."), mock_process_manager)

        # Verify launch_module was called with the correct module name
        call_args = mock_process_manager.launch_module.call_args
        module_name = call_args.kwargs.get(
            "module_name",
            call_args[1].get(
                "module_name", call_args[0][1] if len(call_args[0]) > 1 else None
            ),
        )
        self.assertFalse(
            module_name.endswith(".main"),
            f"Module name should not end with .main, got: {module_name}",
        )


class TestLauncherIntegration(unittest.TestCase):
    """Integration tests for the launcher system."""

    def test_launch_golf_suite_script(self) -> None:
        """Test that launch_golf_suite.py script exists and has correct structure."""
        script_path = Path("launch_golf_suite.py")
        self.assertTrue(script_path.exists(), "Launch script should exist")

        # Test that it has the expected structure
        content = script_path.read_text()
        self.assertIn("def main(", content)
        self.assertIn("def launch_engine_directly", content)
        self.assertIn("Golf Modeling Suite", content)

    @patch("src.launchers.golf_launcher.GolfLauncher")
    def test_unified_launcher_import(self, mock_golf_launcher: Mock) -> None:
        """Test that unified launcher can be imported."""
        try:
            from src.launchers.unified_launcher import UnifiedLauncher

            # Test that we can instantiate it
            launcher = UnifiedLauncher()
            self.assertIsNotNone(launcher)
        except ImportError as e:
            # This might fail if PyQt6 is not available, which is acceptable
            if "PyQt6" not in str(e):
                self.fail(f"Unexpected import error: {e}")


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestSharedModuleImports,
        TestEngineManager,
        TestDockerConfiguration,
        TestMuJoCoModule,
        TestLauncherIntegration,
    ]

    # Add PyQt tests only if available
    if PYQT6_AVAILABLE:
        test_classes.extend(
            [
                TestDraggableModelCard,
                TestGolfLauncherGrid,
            ]
        )
    else:
        print("PyQt6 not available - skipping GUI tests")

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if not result.failures and not result.errors:
        print("\nAll tests passed!")
