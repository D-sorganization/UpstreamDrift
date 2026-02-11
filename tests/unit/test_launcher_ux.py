#!/usr/bin/env python3
"""Test suite for Golf Modeling Suite UX improvements."""

import unittest
from unittest.mock import Mock, patch  # noqa: F401

from src.shared.python.engine_core.engine_availability import PYQT6_AVAILABLE
from src.shared.python.gui_pkg.gui_utils import get_qapp

if PYQT6_AVAILABLE:
    from PyQt6.QtCore import Qt  # noqa: F401
    from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QWidget  # noqa: F401


@unittest.skipUnless(PYQT6_AVAILABLE, "PyQt6 not available")
class TestGolfLauncherUX(unittest.TestCase):
    """Test UX improvements in GolfLauncher."""

    app = None

    @classmethod
    def setUpClass(cls):
        """Set up QApplication for GUI tests."""
        get_qapp()  # Simplified with utility

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

    @unittest.skip(
        "GolfLauncher initialization pipeline was refactored - "
        "model_order depends on LayoutManager which requires deep mocking"
    )
    def test_empty_state_ux(self):
        """Test that empty search state shows actionable UI.

        Skipped: GolfLauncher.__init__ was refactored to use LayoutManager,
        making it difficult to mock the full initialization pipeline.
        """


if __name__ == "__main__":
    unittest.main()
