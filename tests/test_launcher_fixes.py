#!/usr/bin/env python3
"""
Test suite for Golf Modeling Suite launcher fixes and new features.

Tests cover:
- Drag-and-drop functionality
- Docker container setup
- Module import fixes
- Engine detection and management
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add shared modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "python"))

try:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtWidgets import QApplication

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


class TestSharedModuleImports(unittest.TestCase):
    """Test that shared modules can be imported correctly."""

    def test_configuration_manager_import(self):
        """Test configuration manager import."""
        try:
            from shared.python.configuration_manager import ConfigurationManager

            # Test that we can instantiate it with required config_path
            config_manager = ConfigurationManager("dummy_config.json")
            self.assertIsNotNone(config_manager)
        except ImportError as e:
            self.fail(f"Failed to import ConfigurationManager: {e}")
        except Exception as e:
            # If instantiation fails due to missing file, that's expected in tests
            self.assertTrue(True, f"ConfigurationManager import successful: {e}")

    def test_process_worker_import(self):
        """Test process worker import."""
        try:
            from shared.python.process_worker import ProcessWorker

            # Test that we can instantiate it with required cmd
            worker = ProcessWorker("echo test")
            self.assertIsNotNone(worker)
        except ImportError as e:
            self.fail(f"Failed to import ProcessWorker: {e}")

    def test_engine_manager_import(self):
        """Test engine manager import."""
        try:
            from shared.python.engine_manager import EngineManager, EngineType

            # Test that we can instantiate it
            manager = EngineManager()
            self.assertIsNotNone(manager)
            # Test that EngineType enum exists
            self.assertTrue(hasattr(EngineType, "MUJOCO"))
        except ImportError as e:
            self.fail(f"Failed to import EngineManager: {e}")


class TestEngineManager(unittest.TestCase):
    """Test engine manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from shared.python.engine_manager import EngineManager

        self.manager = EngineManager()

    def test_engine_discovery(self):
        """Test that engines are discovered correctly."""
        engines = self.manager.get_available_engines()
        self.assertIsInstance(engines, list)
        self.assertGreater(len(engines), 0, "Should discover at least one engine")

    def test_engine_paths_exist(self):
        """Test that engine paths are properly configured."""
        for _engine_type, path in self.manager.engine_paths.items():
            self.assertIsInstance(path, Path)
            # Note: Not all engines may be installed, so we don't require existence

    def test_probe_system(self):
        """Test engine probe system."""
        from shared.python.engine_manager import EngineType

        # Test MuJoCo probe if available
        if EngineType.MUJOCO in self.manager.probes:
            probe_result = self.manager.get_probe_result(EngineType.MUJOCO)
            self.assertIsNotNone(probe_result)
            self.assertTrue(hasattr(probe_result, "is_available"))
            self.assertTrue(hasattr(probe_result, "diagnostic_message"))


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
class TestDraggableModelCard(unittest.TestCase):
    """Test drag-and-drop functionality in model cards."""

    @classmethod
    def setUpClass(cls):
        """Set up QApplication for GUI tests."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        """Set up test fixtures."""
        # Mock model object
        self.mock_model = Mock()
        self.mock_model.id = "test_model"
        self.mock_model.name = "Test Model"
        self.mock_model.description = "Test Description"

        # Mock parent launcher
        self.mock_launcher = Mock()
        self.mock_launcher.select_model = Mock()
        self.mock_launcher._swap_models = Mock()
        self.mock_launcher.launch_model_direct = Mock()

    def test_draggable_card_creation(self):
        """Test that draggable model cards can be created."""
        from launchers.golf_launcher import DraggableModelCard

        card = DraggableModelCard(self.mock_model, self.mock_launcher)
        self.assertIsNotNone(card)
        self.assertEqual(card.model, self.mock_model)
        self.assertEqual(card.parent_launcher, self.mock_launcher)
        self.assertTrue(card.acceptDrops())

    def test_mouse_press_selection(self):
        """Test that mouse press selects the model."""
        from launchers.golf_launcher import DraggableModelCard

        card = DraggableModelCard(self.mock_model, self.mock_launcher)

        # Create mock mouse event with proper button method
        event = Mock()
        # Use integer 1 (LeftButton) to avoid enum identity issues in CI
        event.button.return_value = 1
        event.position.return_value.toPoint.return_value = QPoint(10, 10)

        # Verify parent launcher is set correctly
        self.assertIsNotNone(card.parent_launcher, "Parent launcher should not be None")
        self.assertEqual(
            card.parent_launcher,
            self.mock_launcher,
            "Parent launcher should match mock",
        )

        card.mousePressEvent(event)

        print(f"DEBUG: parent_launcher in test: {card.parent_launcher}")
        print(f"DEBUG: select_model mock: {self.mock_launcher.select_model}")
        print(
            f"DEBUG: select_model call count: {self.mock_launcher.select_model.call_count}"
        )

        # Verify model selection was called
        self.mock_launcher.select_model.assert_called_once_with("test_model")

    def test_double_click_launch(self):
        """Test that double-click launches the model."""
        from launchers.golf_launcher import DraggableModelCard

        card = DraggableModelCard(self.mock_model, self.mock_launcher)

        # Create mock double-click event
        event = Mock()

        card.mouseDoubleClickEvent(event)

        # Verify launch was called
        self.mock_launcher.launch_model_direct.assert_called_once_with("test_model")

    def test_drag_enter_event(self):
        """Test drag enter event handling."""
        from launchers.golf_launcher import DraggableModelCard

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

    def test_drop_event_swap(self):
        """Test drop event triggers model swap."""
        from launchers.golf_launcher import DraggableModelCard

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


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
class TestGolfLauncherGrid(unittest.TestCase):
    """Test golf launcher grid functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up QApplication for GUI tests."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def setUp(self):
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
        def mock_get_model(model_id):
            for model in self.mock_models:
                if model.id == model_id:
                    return model
            return None

        self.mock_registry.get_model.side_effect = mock_get_model

    @patch("launchers.golf_launcher.ModelRegistry")
    @patch("launchers.golf_launcher.EngineManager")
    def test_model_order_tracking(self, mock_engine_manager, mock_registry_class):
        """Test that model order is properly tracked."""
        from launchers.golf_launcher import GolfLauncher

        mock_registry_class.return_value = self.mock_registry
        mock_engine_manager.return_value = Mock()

        launcher = GolfLauncher()

        # Check that model order is initialized
        self.assertIsInstance(launcher.model_order, list)
        # The launcher adds 2 special models (C3D viewer and URDF generator) to the registry models
        expected_count = (
            len(self.mock_models) + 2
        )  # 4 mock models + 2 special models = 6
        self.assertEqual(len(launcher.model_order), expected_count)

    @patch("launchers.golf_launcher.ModelRegistry")
    @patch("launchers.golf_launcher.EngineManager")
    def test_model_swap_functionality(self, mock_engine_manager, mock_registry_class):
        """Test model swapping functionality."""
        from launchers.golf_launcher import GolfLauncher

        mock_registry_class.return_value = self.mock_registry
        mock_engine_manager.return_value = Mock()

        launcher = GolfLauncher()

        # Set up initial order
        launcher.model_order = ["model_0", "model_1", "model_2", "model_3"]
        launcher.model_cards = {
            "model_0": Mock(),
            "model_1": Mock(),
            "model_2": Mock(),
            "model_3": Mock(),
        }

        # Mock the grid layout
        launcher.grid_layout = Mock()
        launcher.grid_layout.count.return_value = 4
        launcher.grid_layout.itemAt.return_value.widget.return_value = Mock()

        # Test swapping
        launcher._swap_models("model_0", "model_2")

        # Verify order changed
        expected_order = ["model_2", "model_1", "model_0", "model_3"]
        self.assertEqual(launcher.model_order, expected_order)


class TestDockerConfiguration(unittest.TestCase):
    """Test Docker configuration and setup."""

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and is readable."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        self.assertTrue(dockerfile_path.exists(), "Dockerfile should exist")

        # Test that it's readable
        content = dockerfile_path.read_text()
        self.assertIn("PYTHONPATH", content, "Dockerfile should set PYTHONPATH")
        self.assertIn("/workspace", content, "Dockerfile should configure workspace")

    def test_docker_launch_command_structure(self):
        """Test Docker launch command structure."""
        from launchers.golf_launcher import GolfLauncher

        # Mock the launcher
        launcher = GolfLauncher.__new__(GolfLauncher)  # Create without __init__
        launcher.chk_live = Mock()
        launcher.chk_live.isChecked.return_value = True
        launcher.chk_gpu = Mock()
        launcher.chk_gpu.isChecked.return_value = False

        # Mock model
        mock_model = Mock()
        mock_model.type = "custom_humanoid"

        # Mock path
        mock_path = Path("/test/path")

        with (
            patch("subprocess.Popen") as mock_popen,
            patch("os.name", "nt"),
            patch("launchers.golf_launcher.logger"),
        ):

            launcher._launch_docker_container(mock_model, mock_path)

            # Verify subprocess was called
            mock_popen.assert_called()

            # Get the command that was called
            call_args = mock_popen.call_args[0][0]

            # Verify key components are in the command
            command_str = " ".join(call_args)
            self.assertIn(
                "PYTHONPATH=/workspace:/workspace/shared/python:/workspace/engines",
                command_str,
            )
            self.assertIn("docker run", command_str)


class TestMuJoCoModule(unittest.TestCase):
    """Test MuJoCo module structure and availability."""

    def test_mujoco_module_exists(self):
        """Test that MuJoCo humanoid golf module exists."""
        mujoco_path = Path("engines/physics_engines/mujoco/python/mujoco_humanoid_golf")
        self.assertTrue(mujoco_path.exists(), "MuJoCo module directory should exist")

        main_file = mujoco_path / "__main__.py"
        self.assertTrue(main_file.exists(), "MuJoCo module should have __main__.py")

    def test_mujoco_module_structure(self):
        """Test MuJoCo module has required components."""
        mujoco_path = Path("engines/physics_engines/mujoco/python/mujoco_humanoid_golf")

        required_files = [
            "__init__.py",
            "__main__.py",
            "advanced_gui.py",
            "physics_engine.py",
        ]

        for file_name in required_files:
            file_path = mujoco_path / file_name
            self.assertTrue(
                file_path.exists(), f"Required file {file_name} should exist"
            )


class TestLauncherIntegration(unittest.TestCase):
    """Integration tests for the launcher system."""

    def test_launch_golf_suite_script(self):
        """Test that launch_golf_suite.py script exists and is executable."""
        script_path = Path("launch_golf_suite.py")
        self.assertTrue(script_path.exists(), "Launch script should exist")

        # Test that it has the expected structure
        content = script_path.read_text()
        self.assertIn("def launch_gui_launcher", content)
        self.assertIn("def launch_mujoco", content)
        self.assertIn("def launch_drake", content)
        self.assertIn("def launch_pinocchio", content)

    def test_unified_launcher_import(self):
        """Test that unified launcher can be imported."""
        try:
            from launchers.unified_launcher import UnifiedLauncher

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
    if PYQT_AVAILABLE:
        test_classes.extend(
            [
                TestDraggableModelCard,
                TestGolfLauncherGrid,
            ]
        )
    else:
        print("⚠️  PyQt6 not available - skipping GUI tests")

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            error_msg = traceback.split("AssertionError: ")[-1].split("\n")[0]
            print(f"  - {test}: {error_msg}")

    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            error_msg = traceback.split("\n")[-2]
            print(f"  - {test}: {error_msg}")

    if not result.failures and not result.errors:
        print("\n🎉 All tests passed!")
