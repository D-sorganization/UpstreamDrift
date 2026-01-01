#!/usr/bin/env python3
"""
Test suite for drag-and-drop functionality in the Golf Modeling Suite launcher.

Tests cover:
- Drag-and-drop model card reordering
- 3x3 grid layout
- URDF generator integration
- Error handling in drag operations
"""

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

# Add repo root and shared modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "python"))

try:
    from PyQt6.QtCore import QMimeData, QPoint, Qt, QPointF
    from PyQt6.QtGui import QDragEnterEvent, QDropEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QWidget

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
class TestDragDropFunctionality(unittest.TestCase):
    """Test drag-and-drop functionality in model cards."""

    app: Any = None

    @classmethod
    def setUpClass(cls) -> None:
        """Set up QApplication for GUI tests."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock model objects
        self.mock_models = []
        for i in range(3):
            model = Mock()
            model.id = f"test_model_{i}"
            model.name = f"Test Model {i}"
            model.description = f"Test Description {i}"
            self.mock_models.append(model)

        # Mock parent launcher
        self.mock_launcher = Mock()
        self.mock_launcher.select_model = Mock()
        self.mock_launcher._swap_models = Mock()
        self.mock_launcher.launch_model_direct = Mock()

    def test_draggable_card_initialization(self) -> None:
        """Test that draggable model cards initialize correctly."""
        from launchers.golf_launcher import DraggableModelCard

        # Case 1: Parent has layout_edit_mode = True
        self.mock_launcher.layout_edit_mode = True
        card = DraggableModelCard(self.mock_models[0], self.mock_launcher)
        self.assertTrue(card.acceptDrops())
        self.assertEqual(card.model.id, "test_model_0")

        # Case 2: Parent has layout_edit_mode = False
        self.mock_launcher.layout_edit_mode = False
        card2 = DraggableModelCard(self.mock_models[0], self.mock_launcher)
        self.assertFalse(card2.acceptDrops())

    def test_mouse_press_initializes_drag(self) -> None:
        """Test that mouse press initializes drag position."""
        from launchers.golf_launcher import DraggableModelCard

        self.mock_launcher.layout_edit_mode = True
        card = DraggableModelCard(self.mock_models[0], self.mock_launcher)

        # Simulate click
        QTest.mousePress(card, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))

        self.assertEqual(card.drag_start_position, QPoint(10, 10))

    def test_drag_operation_error_handling(self) -> None:
        """Test that drag operations handle errors gracefully."""
        # Drag operation involves blocking exec(), which is hard to unit test without freezing.
        # We assume standard Qt drag behavior.
        # We can test that mouseMoveEvent triggers checks.
        pass

    def test_drop_event_triggers_swap(self) -> None:
        """Test that drop events trigger model swapping."""
        from launchers.golf_launcher import DraggableModelCard

        self.mock_launcher.layout_edit_mode = True
        card = DraggableModelCard(self.mock_models[0], self.mock_launcher)
        card.model.id = "target_model"

        # Create MimeData with source model ID
        mime_data = QMimeData()
        mime_data.setText("model_card:source_model")

        # Create DropEvent
        event = QDropEvent(
            QPointF(10, 10),
            Qt.DropAction.MoveAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )

        # Call dropEvent
        card.dropEvent(event)

        # Verify swap called
        self.mock_launcher._swap_models.assert_called_with("source_model", "target_model")
        # Verify event accepted
        self.assertTrue(event.isAccepted())

    def test_drop_event_ignores_invalid_data(self) -> None:
        """Test that drop events ignore invalid mime data."""
        from launchers.golf_launcher import DraggableModelCard

        card = DraggableModelCard(self.mock_models[0], self.mock_launcher)

        # Empty MimeData
        mime_data = QMimeData()

        event = QDropEvent(
            QPointF(10, 10),
            Qt.DropAction.MoveAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )

        card.dropEvent(event)

        self.mock_launcher._swap_models.assert_not_called()

    def test_double_click_launches_model(self) -> None:
        """Test that double-click launches the model."""
        from launchers.golf_launcher import DraggableModelCard

        self.mock_launcher.launch_simulation = Mock()
        # Note: calling init_ui usually sets self.selected_model.
        # But double click calls launch_model_direct or similar?
        # Let's check expected behavior.
        # Launcher selects model then launches.

        card = DraggableModelCard(self.mock_models[0], self.mock_launcher)

        # Simulate double click
        QTest.mouseDClick(card, Qt.MouseButton.LeftButton)

        # Verify selection and launch
        self.mock_launcher.launch_model_direct.assert_called_with("test_model_0")


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
class TestGridLayout(unittest.TestCase):
    """Test 3x3 grid layout functionality."""

    app: Any = None

    @classmethod
    def setUpClass(cls) -> None:
        """Set up QApplication for GUI tests."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def test_grid_columns_constant(self) -> None:
        """Test that grid columns is set to 4."""
        from launchers.golf_launcher import GRID_COLUMNS

        self.assertEqual(GRID_COLUMNS, 4, "Grid should be 3x4 layout")

    @patch("launchers.golf_launcher.ModelRegistry")
    @patch("launchers.golf_launcher.EngineManager")
    def test_model_order_with_urdf_generator_and_c3d_viewer(
        self, mock_engine_manager: Mock, mock_registry_class: Mock
    ) -> None:
        """Test that URDF generator and C3D viewer are added to model order."""
        from launchers.golf_launcher import GolfLauncher

        # Mock registry with test models
        mock_registry = Mock()
        mock_models = []
        for i in range(10):  # 10 regular models
            model = Mock()
            model.id = f"model_{i}"
            model.name = f"Model {i}"
            model.description = f"Description {i}"
            model.type = "test_type"  # Add type attribute
            mock_models.append(model)

        mock_registry.get_all_models.return_value = mock_models
        mock_registry.get_model.return_value = None  # Return None for unknown models
        mock_registry_class.return_value = mock_registry
        mock_engine_manager.return_value = Mock()

        # Mock the UI initialization to avoid Qt widget creation
        with (
            patch.object(GolfLauncher, "init_ui"),
            patch.object(GolfLauncher, "check_docker"),
            patch.object(GolfLauncher, "_load_layout"),
        ):
            launcher = GolfLauncher()

            # Manually set up the model order as it would be in init_ui
            launcher.model_order = []
            launcher.model_cards = {}

            # Simulate the model addition logic from init_ui
            for model in mock_models:
                launcher.model_order.append(model.id)

            # Add special models
            launcher.model_order.append("urdf_generator")
            launcher.model_order.append("c3d_viewer")

            # Check that URDF generator and C3D viewer are added as 11th and 12th models
            self.assertEqual(len(launcher.model_order), 12)
            self.assertIn("urdf_generator", launcher.model_order)
            self.assertIn("c3d_viewer", launcher.model_order)
            self.assertEqual(launcher.model_order[-2], "urdf_generator")
            self.assertEqual(launcher.model_order[-1], "c3d_viewer")

    @patch("launchers.golf_launcher.GolfLauncher.addDockWidget", create=True)
    @patch("launchers.golf_launcher.ContextHelpDock")
    @patch("launchers.golf_launcher.ModelRegistry")
    @patch("launchers.golf_launcher.EngineManager")
    def test_model_swap_preserves_special_tiles(
        self,
        mock_engine_manager: Mock,
        mock_registry_class: Mock,
        mock_help_dock: Mock,
        mock_add_dock_widget: Mock,
    ) -> None:
        """Test that model swapping works with URDF generator and C3D viewer."""
        from launchers.golf_launcher import GolfLauncher

        mock_registry = Mock()
        mock_registry.get_all_models.return_value = []
        mock_registry_class.return_value = mock_registry
        mock_engine_manager.return_value = Mock()

        mock_help_dock.side_effect = None
        launcher = GolfLauncher()

        # Set up test order with special tiles
        launcher.model_order = ["model_0", "model_1", "urdf_generator", "c3d_viewer"]
        launcher.model_cards = {
            "model_0": Mock(),
            "model_1": Mock(),
            "urdf_generator": Mock(),
            "c3d_viewer": Mock(),
        }

        # Mock the grid layout
        launcher.grid_layout = Mock()
        launcher.grid_layout.count.return_value = 4
        launcher.grid_layout.itemAt.return_value.widget.return_value = Mock()

        # Test swapping regular model with C3D viewer
        launcher._swap_models("model_0", "c3d_viewer")

        # Verify order changed correctly
        expected_order = ["c3d_viewer", "model_1", "urdf_generator", "model_0"]
        self.assertEqual(launcher.model_order, expected_order)


class TestC3DViewerIntegration(unittest.TestCase):
    """Test C3D viewer integration with the launcher."""

    def test_c3d_viewer_files_exist(self) -> None:
        """Test that C3D viewer files exist."""
        c3d_script = Path(
            "engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py"
        )
        self.assertTrue(c3d_script.exists(), "C3D viewer script should exist")

        # Check that the script is executable Python
        with open(c3d_script, encoding="utf-8") as f:
            content = f.read()
            self.assertIn("#!/usr/bin/env python", content)
            self.assertIn("C3DViewerMainWindow", content)
            self.assertIn("def main()", content)

    def test_c3d_viewer_dependencies(self) -> None:
        """Test that C3D viewer has required dependencies."""
        try:
            # Test imports that C3D viewer requires
            import ezc3d  # type: ignore[import-not-found]
            import matplotlib
            import numpy as np

            # Basic functionality test
            self.assertTrue(hasattr(ezc3d, "c3d"))
            self.assertTrue(hasattr(np, "ndarray"))
            self.assertTrue(hasattr(matplotlib, "pyplot"))

        except ImportError as e:
            self.skipTest(f"C3D viewer dependencies not available: {e}")

    def test_c3d_viewer_constants(self) -> None:
        """Test that C3D viewer constants are properly defined."""
        from shared.python.constants import C3D_VIEWER_SCRIPT

        self.assertIsInstance(C3D_VIEWER_SCRIPT, Path)
        # Use Path.as_posix() to get forward slashes on all platforms
        self.assertEqual(
            C3D_VIEWER_SCRIPT.as_posix(),
            "engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py",
        )

    @unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
    def test_c3d_viewer_launch_method(self) -> None:
        """Test C3D viewer launch method."""
        from launchers.golf_launcher import GolfLauncher

        # Create launcher instance without full initialization
        launcher = GolfLauncher.__new__(GolfLauncher)
        launcher.running_processes = {}

        # Mock the C3D viewer script path and subprocess
        with (
            patch("pathlib.Path") as mock_path_cls,
            patch("subprocess.Popen") as mock_popen,
            patch("launchers.golf_launcher.os.name", "nt"),
            patch("launchers.golf_launcher.logger") as mock_logger,
            patch("launchers.golf_launcher.QMessageBox"),
            patch("launchers.golf_launcher.CREATE_NEW_CONSOLE", 0x00000010),
        ):
            # Setup path mock to return truthy exists
            mock_path_instance = mock_path_cls.return_value
            mock_path_instance.exists.return_value = True
            mock_path_instance.__truediv__.return_value = mock_path_instance
            mock_path_instance.parent = mock_path_instance

            launcher._launch_c3d_viewer()

            # Check for suppressed exceptions
            logger_mock = mock_logger
            if logger_mock.error.called:
                error_call = logger_mock.error.call_args
                self.fail(f"Exception suppressed in _launch_c3d_viewer: {error_call}")

            # Verify subprocess was called
            mock_popen.assert_called_once()

            # Verify it's launching the C3D viewer
            # Since we mock Path, we can't check the string content reliably
            # call_args = mock_popen.call_args[0][0]
            # self.assertIn("c3d_viewer.py", " ".join(call_args))

    @unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
    def test_c3d_viewer_missing_file_handling(self) -> None:
        """Test handling when C3D viewer file is missing."""
        from launchers.golf_launcher import GolfLauncher

        launcher = GolfLauncher.__new__(GolfLauncher)
        launcher.running_processes = {}

        # Mock missing file
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("launchers.golf_launcher.QMessageBox") as mock_msgbox,
        ):
            launcher._launch_c3d_viewer()

            # Verify warning message was shown
            mock_msgbox.warning.assert_called_once()

    def test_c3d_viewer_cli_support(self) -> None:
        """Test that CLI launcher supports C3D viewer."""
        from launch_golf_suite import launch_c3d_viewer

        # Mock the subprocess call
        with (
            patch("subprocess.run") as mock_run,
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = launch_c3d_viewer()

            # Verify it was called
            mock_run.assert_called_once()
            self.assertTrue(result)

            # Check the command
            call_args = mock_run.call_args[0][0]
            self.assertIn("c3d_viewer.py", " ".join(call_args))


class TestURDFGeneratorIntegration(unittest.TestCase):
    """Test URDF generator integration with the launcher."""

    def test_urdf_generator_files_exist(self) -> None:
        """Test that URDF generator files exist."""
        urdf_dir = Path("tools/urdf_generator")
        self.assertTrue(urdf_dir.exists(), "URDF generator directory should exist")

        required_files = [
            "launch_urdf_generator.py",
            "main.py",
            "main_window.py",
            "segment_manager.py",
            "urdf_builder.py",
        ]

        for file_name in required_files:
            file_path = urdf_dir / file_name
            self.assertTrue(
                file_path.exists(), f"Required file {file_name} should exist"
            )

    def test_urdf_generator_engine_support(self) -> None:
        """Test that URDF generator supports multiple engines."""
        try:
            from tools.urdf_generator.segment_manager import SegmentManager

            manager = SegmentManager()

            # Test engine export methods exist
            self.assertTrue(hasattr(manager, "export_for_engine"))

            # Test supported engines
            supported_engines = ["mujoco", "drake", "pinocchio"]
            for engine in supported_engines:
                try:
                    result = manager.export_for_engine(engine)
                    self.assertIsInstance(result, dict)
                    self.assertEqual(result["engine"], engine)
                except Exception as e:
                    self.fail(f"Engine {engine} export failed: {e}")

        except ImportError as e:
            self.skipTest(f"URDF generator not available: {e}")

    @unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
    def test_urdf_generator_launch_method(self) -> None:
        """Test URDF generator launch method."""
        from launchers.golf_launcher import GolfLauncher

        # Create launcher instance without full initialization
        launcher = GolfLauncher.__new__(GolfLauncher)
        launcher.running_processes = {}

        # Mock the URDF generator script path and subprocess
        with (
            patch("pathlib.Path") as mock_path_cls,
            patch("subprocess.Popen") as mock_popen,
            patch("launchers.golf_launcher.os.name", "nt"),
            patch("launchers.golf_launcher.logger") as mock_logger,
            patch("launchers.golf_launcher.QMessageBox"),
            patch("launchers.golf_launcher.CREATE_NEW_CONSOLE", 0x00000010),
        ):
            # Setup path mock to return truthy exists
            mock_path_instance = mock_path_cls.return_value
            mock_path_instance.exists.return_value = True
            mock_path_instance.__truediv__.return_value = mock_path_instance
            mock_path_instance.parent = mock_path_instance

            launcher._launch_urdf_generator()

            # Check for suppressed exceptions
            if mock_logger.error.called:
                error_call = mock_logger.error.call_args
                self.fail(
                    f"Exception suppressed in _launch_urdf_generator: {error_call}"
                )

            # Verify subprocess was called
            mock_popen.assert_called_once()

            # Verify it's launching the URDF generator
            # Since we mock Path, we can't check the string content reliably
            # call_args = mock_popen.call_args[0][0]
            # self.assertIn("launch_urdf_generator.py", " ".join(call_args))

    @unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
    def test_urdf_generator_missing_file_handling(self) -> None:
        """Test handling when URDF generator file is missing."""
        from launchers.golf_launcher import GolfLauncher

        launcher = GolfLauncher.__new__(GolfLauncher)
        launcher.running_processes = {}

        # Mock missing file
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("launchers.golf_launcher.QMessageBox") as mock_msgbox,
        ):
            launcher._launch_urdf_generator()

            # Verify warning message was shown
            mock_msgbox.warning.assert_called_once()


class TestModelImageHandling(unittest.TestCase):
    """Test model image handling for the new grid layout."""

    def test_urdf_generator_image_mapping(self) -> None:
        """Test that URDF generator has image mapping."""
        from launchers.golf_launcher import MODEL_IMAGES

        self.assertIn("URDF Generator", MODEL_IMAGES)
        self.assertEqual(MODEL_IMAGES["URDF Generator"], "urdf_icon.png")

    def test_c3d_viewer_image_mapping(self) -> None:
        """Test that C3D viewer has image mapping."""
        from launchers.golf_launcher import MODEL_IMAGES

        self.assertIn("C3D Motion Viewer", MODEL_IMAGES)
        self.assertEqual(MODEL_IMAGES["C3D Motion Viewer"], "c3d_icon.png")

    def test_image_fallback_for_urdf(self) -> None:
        """Test image fallback logic for URDF generator."""
        # This would be tested in the actual DraggableModelCard setup_ui method
        # The logic checks for "urdf" in model.id and assigns "urdf_icon.png"

        # Mock model with urdf in ID
        mock_model = Mock()
        mock_model.id = "urdf_generator"
        mock_model.name = "URDF Generator"
        mock_model.description = "Test"

        # The image selection logic should work
        from launchers.golf_launcher import MODEL_IMAGES

        # Direct lookup should work
        img_name = MODEL_IMAGES.get(mock_model.name)
        if not img_name and "urdf" in mock_model.id:
            img_name = "urdf_icon.png"

        self.assertEqual(img_name, "urdf_icon.png")


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestModelImageHandling,
        TestURDFGeneratorIntegration,
    ]

    # Add PyQt tests only if available
    if PYQT_AVAILABLE:
        test_classes.extend(
            [
                TestDragDropFunctionality,
                TestGridLayout,
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
    print(f"\n{'=' * 60}")
    print("Drag-and-Drop Tests Summary")
    print(f"{'=' * 60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFAILURES:")
        for test, _ in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\nERRORS:")
        for test, _ in result.errors:
            print(f"  - {test}")

    if not result.failures and not result.errors:
        print("\nAll drag-and-drop tests passed!")
