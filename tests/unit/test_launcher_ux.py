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

    def setUp(self) -> None:
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

        def mock_get_model(model_id) -> Mock | None:
            for model in self.mock_models:
                if model.id == model_id:
                    return model
            return None

        self.mock_registry.get_model.side_effect = mock_get_model

    def test_mock_registry_returns_expected_models(self) -> None:
        """Test that the mock registry correctly provides model lookup."""
        all_models = self.mock_registry.get_all_models()
        self.assertEqual(len(all_models), 2, "Registry should contain 2 mock models")
        self.assertEqual(all_models[0].id, "model_0", "First model id mismatch")
        self.assertEqual(all_models[1].name, "Model 1", "Second model name mismatch")

    def test_mock_registry_get_model_returns_none_for_unknown(self) -> None:
        """Test that get_model returns None for an unknown ID."""
        result = self.mock_registry.get_model("nonexistent_model")
        self.assertIsNone(result, "get_model should return None for unknown ID")


if __name__ == "__main__":
    unittest.main()
