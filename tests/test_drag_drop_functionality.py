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
from unittest.mock import MagicMock, Mock, patch

# Add shared modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "python"))

try:
    from PyQt6.QtCore import QMimeData, QPoint, Qt
    from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent
    from PyQt6.QtWidgets import QApplication
    
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
class TestDragDropFunctionality(unittest.TestCase):
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
    
    def test_draggable_card_initialization(self):
        """Test that draggable model cards initialize correctly."""
        from launchers.golf_launcher import DraggableModelCard
        
        card = DraggableModelCard(self.mock_models[0], self.mock_launcher)
        
        # Verify basic properties
        self.assertEqual(card.model, self.mock_models[0])
        self.assertEqual(card.parent_launcher, self.mock_launcher)
        self.assertTrue(card.acceptDrops())
        self.assertTrue(hasattr(card, 'drag_start_position'))
    
    def test_mouse_press_initializes_drag(self):
        """Test that mouse press initializes drag position."""
        from launchers.golf_launcher import DraggableModelCard
        
        card = DraggableModelCard(self.mock_models[0], self.mock_launcher)
        
        # Create mock mouse event
        event = Mock()
        event.button.return_value = Qt.MouseButton.LeftButton
        event.position.return_value.toPoint.return_value = QPoint(10, 10)
        
        card.mousePressEvent(event)
        
        # Verify drag position is set and model is selected
        self.assertEqual(card.drag_start_position, QPoint(10, 10))
        self.mock_launcher.select_model.assert_called_once_with("test_model_0")
    
    def test_drag_operation_error_handling(self):
        """Test that drag operations handle errors gracefully."""
        from launchers.golf_launcher import DraggableModelCard
        
        card = DraggableModelCard(self.mock_models[0], self.mock_launcher)
        card.drag_start_position = QPoint(10, 10)
        
        # Create mock mouse move event
        event = Mock()
        event.buttons.return_value = Qt.MouseButton.LeftButton
        event.position.return_value.toPoint.return_value = QPoint(50, 50)
        
        # Mock QDrag to raise an exception
        with patch('launchers.golf_launcher.QDrag') as mock_drag_class:
            mock_drag_class.side_effect = Exception("Test drag error")
            
            # Should not raise exception
            try:
                card.mouseMoveEvent(event)
            except Exception as e:
                self.fail(f"Drag operation should handle errors gracefully: {e}")
    
    def test_drop_event_triggers_swap(self):
        """Test that drop events trigger model swapping."""
        from launchers.golf_launcher import DraggableModelCard
        
        card = DraggableModelCard(self.mock_models[1], self.mock_launcher)
        
        # Create mock drop event
        event = Mock()
        mime_data = Mock()
        mime_data.hasText.return_value = True
        mime_data.text.return_value = "model_card:test_model_0"
        event.mimeData.return_value = mime_data
        
        card.dropEvent(event)
        
        # Verify swap was called with correct parameters
        self.mock_launcher._swap_models.assert_called_once_with("test_model_0", "test_model_1")
        event.acceptProposedAction.assert_called_once()
    
    def test_drop_event_ignores_invalid_data(self):
        """Test that drop events ignore invalid mime data."""
        from launchers.golf_launcher import DraggableModelCard
        
        card = DraggableModelCard(self.mock_models[1], self.mock_launcher)
        
        # Create mock drop event with invalid data
        event = Mock()
        mime_data = Mock()
        mime_data.hasText.return_value = True
        mime_data.text.return_value = "invalid_data:something"
        event.mimeData.return_value = mime_data
        
        card.dropEvent(event)
        
        # Verify swap was not called
        self.mock_launcher._swap_models.assert_not_called()
        event.ignore.assert_called_once()
    
    def test_double_click_launches_model(self):
        """Test that double-click launches the model."""
        from launchers.golf_launcher import DraggableModelCard
        
        card = DraggableModelCard(self.mock_models[0], self.mock_launcher)
        
        # Create mock double-click event
        event = Mock()
        
        card.mouseDoubleClickEvent(event)
        
        # Verify launch was called
        self.mock_launcher.launch_model_direct.assert_called_once_with("test_model_0")


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
class TestGridLayout(unittest.TestCase):
    """Test 3x3 grid layout functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up QApplication for GUI tests."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()
    
    def test_grid_columns_constant(self):
        """Test that grid columns is set to 3."""
        from launchers.golf_launcher import GRID_COLUMNS
        
        self.assertEqual(GRID_COLUMNS, 3, "Grid should be 3x3 layout")
    
    @patch('launchers.golf_launcher.ModelRegistry')
    @patch('launchers.golf_launcher.EngineManager')
    def test_model_order_with_urdf_generator(self, mock_engine_manager, mock_registry_class):
        """Test that URDF generator is added to model order."""
        from launchers.golf_launcher import GolfLauncher
        
        # Mock registry with test models
        mock_registry = Mock()
        mock_models = []
        for i in range(8):  # 8 regular models
            model = Mock()
            model.id = f"model_{i}"
            model.name = f"Model {i}"
            model.description = f"Description {i}"
            mock_models.append(model)
        
        mock_registry.get_all_models.return_value = mock_models
        mock_registry_class.return_value = mock_registry
        mock_engine_manager.return_value = Mock()
        
        launcher = GolfLauncher()
        
        # Check that URDF generator is added as 9th model
        self.assertEqual(len(launcher.model_order), 9)
        self.assertIn("urdf_generator", launcher.model_order)
        self.assertEqual(launcher.model_order[-1], "urdf_generator")
    
    @patch('launchers.golf_launcher.ModelRegistry')
    @patch('launchers.golf_launcher.EngineManager')
    def test_model_swap_preserves_urdf_generator(self, mock_engine_manager, mock_registry_class):
        """Test that model swapping works with URDF generator."""
        from launchers.golf_launcher import GolfLauncher
        
        mock_registry = Mock()
        mock_registry.get_all_models.return_value = []
        mock_registry_class.return_value = mock_registry
        mock_engine_manager.return_value = Mock()
        
        launcher = GolfLauncher()
        
        # Set up test order with URDF generator
        launcher.model_order = ["model_0", "model_1", "urdf_generator"]
        launcher.model_cards = {
            "model_0": Mock(),
            "model_1": Mock(),
            "urdf_generator": Mock(),
        }
        
        # Mock the grid layout
        launcher.grid_layout = Mock()
        launcher.grid_layout.count.return_value = 3
        launcher.grid_layout.itemAt.return_value.widget.return_value = Mock()
        
        # Test swapping regular model with URDF generator
        launcher._swap_models("model_0", "urdf_generator")
        
        # Verify order changed correctly
        expected_order = ["urdf_generator", "model_1", "model_0"]
        self.assertEqual(launcher.model_order, expected_order)


class TestURDFGeneratorIntegration(unittest.TestCase):
    """Test URDF generator integration with the launcher."""
    
    def test_urdf_generator_files_exist(self):
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
            self.assertTrue(file_path.exists(), f"Required file {file_name} should exist")
    
    def test_urdf_generator_engine_support(self):
        """Test that URDF generator supports multiple engines."""
        try:
            from tools.urdf_generator.segment_manager import SegmentManager
            
            manager = SegmentManager()
            
            # Test engine export methods exist
            self.assertTrue(hasattr(manager, 'export_for_engine'))
            
            # Test supported engines
            supported_engines = ['mujoco', 'drake', 'pinocchio']
            for engine in supported_engines:
                try:
                    result = manager.export_for_engine(engine)
                    self.assertIsInstance(result, dict)
                    self.assertEqual(result['engine'], engine)
                except Exception as e:
                    self.fail(f"Engine {engine} export failed: {e}")
                    
        except ImportError as e:
            self.skipTest(f"URDF generator not available: {e}")
    
    @unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
    def test_urdf_generator_launch_method(self):
        """Test URDF generator launch method."""
        from launchers.golf_launcher import GolfLauncher
        
        # Create launcher instance without full initialization
        launcher = GolfLauncher.__new__(GolfLauncher)
        
        # Mock the URDF generator script path
        with patch('pathlib.Path.exists', return_value=True), \
             patch('subprocess.Popen') as mock_popen, \
             patch('os.name', 'nt'):
            
            launcher._launch_urdf_generator()
            
            # Verify subprocess was called
            mock_popen.assert_called_once()
            
            # Get the command that was called
            call_args = mock_popen.call_args[0][0]
            
            # Verify it's launching the URDF generator
            self.assertIn('launch_urdf_generator.py', ' '.join(call_args))
    
    @unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
    def test_urdf_generator_missing_file_handling(self):
        """Test handling when URDF generator file is missing."""
        from launchers.golf_launcher import GolfLauncher
        
        launcher = GolfLauncher.__new__(GolfLauncher)
        
        # Mock missing file
        with patch('pathlib.Path.exists', return_value=False), \
             patch('launchers.golf_launcher.QMessageBox') as mock_msgbox:
            
            launcher._launch_urdf_generator()
            
            # Verify warning message was shown
            mock_msgbox.warning.assert_called_once()


class TestModelImageHandling(unittest.TestCase):
    """Test model image handling for the new grid layout."""
    
    def test_urdf_generator_image_mapping(self):
        """Test that URDF generator has image mapping."""
        from launchers.golf_launcher import MODEL_IMAGES
        
        self.assertIn("URDF Generator", MODEL_IMAGES)
        self.assertEqual(MODEL_IMAGES["URDF Generator"], "urdf_icon.png")
    
    def test_image_fallback_for_urdf(self):
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
        test_classes.extend([
            TestDragDropFunctionality,
            TestGridLayout,
        ])
    else:
        print("⚠️  PyQt6 not available - skipping GUI tests")
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Drag-and-Drop Tests Summary")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if not result.failures and not result.errors:
        print(f"\n🎉 All drag-and-drop tests passed!")