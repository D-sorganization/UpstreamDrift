#!/usr/bin/env python3
"""
Test suite for Golf Modeling Suite UX improvements.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch  # noqa: F401

# Add repo root to sys.path to allow importing modules
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from PyQt6.QtCore import Qt  # noqa: F401
    from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QWidget  # noqa: F401
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
class TestGolfLauncherUX(unittest.TestCase):
    """Test UX improvements in GolfLauncher."""

    app = None

    @classmethod
    def setUpClass(cls):
        """Set up QApplication for GUI tests."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        """Set up test fixtures."""
        self.mock_registry = Mock()
        self.mock_models = []
        for i in range(2):
            model = Mock()
            model.id = f"model_{i}"
            model.name = f"Model {i}"
            model.description = f"Description {i}"
            model.type = "mujoco"
            self.mock_models.append(model)

        self.mock_registry.get_all_models.return_value = self.mock_models
        self.mock_registry.__iter__ = lambda x: iter(self.mock_models)

        def mock_get_model(model_id):
            for model in self.mock_models:
                if model.id == model_id:
                    return model
            return None
        self.mock_registry.get_model.side_effect = mock_get_model

    @patch("launchers.golf_launcher.GolfLauncher._load_layout")
    @patch("launchers.golf_launcher.GolfLauncher.addDockWidget", create=True)
    @patch("launchers.golf_launcher.ContextHelpDock")
    @patch("launchers.golf_launcher._lazy_load_model_registry")
    @patch("launchers.golf_launcher._lazy_load_engine_manager")
    def test_empty_state_ux(
        self,
        mock_lazy_load_engine_manager,
        mock_lazy_load_model_registry,
        mock_help_dock,
        mock_add_dock_widget,
        mock_load_layout,
    ):
        """Test that empty search state shows actionable UI."""
        from launchers.golf_launcher import GolfLauncher

        # Mock lazy loaders
        mock_EM = Mock()
        mock_EM_class = Mock()
        mock_EM_class.return_value = mock_EM
        mock_lazy_load_engine_manager.return_value = (mock_EM_class, Mock())

        mock_MR_class = Mock()
        mock_MR_class.return_value = self.mock_registry
        mock_lazy_load_model_registry.return_value = mock_MR_class

        mock_help_dock.side_effect = None

        launcher = GolfLauncher()

        # Verify initial state (all models visible)
        self.assertGreater(len(launcher.model_order), 0)

        # Simulate searching for something that doesn't exist
        launcher.search_input.setText("nonexistent_model_xyz")
        # Ensure the filter update is triggered
        launcher.update_search_filter("nonexistent_model_xyz")

        # Find widgets in grid layout
        found_empty_widget = None
        for i in range(launcher.grid_layout.count()):
            item = launcher.grid_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                # We are looking for a QWidget that contains a QLabel and QPushButton
                children = widget.findChildren(QPushButton)
                if children:
                    for btn in children:
                        if btn.text() == "Clear Search":
                            found_empty_widget = widget
                            break

        self.assertIsNotNone(found_empty_widget, "Should find empty state widget with Clear Search button")

        # Verify the button functionality
        btn = found_empty_widget.findChild(QPushButton)
        self.assertEqual(btn.text(), "Clear Search")

        # Verify clicking it clears the search
        btn.click()

        # Check if search text is cleared
        self.assertEqual(launcher.search_input.text(), "")
        # Check if filter text is cleared
        self.assertEqual(launcher.current_filter_text, "")

if __name__ == "__main__":
    unittest.main()
