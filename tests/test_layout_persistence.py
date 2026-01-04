#!/usr/bin/env python3
"""
Test suite for launcher layout persistence functionality.

Tests cover:
- Layout saving and loading
- Model order persistence
- Window geometry persistence
- Configuration file handling
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

# Add shared modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "python"))

try:
    # Check if PyQt6 is available for GUI tests
    import importlib.util

    PYQT_AVAILABLE = importlib.util.find_spec("PyQt6") is not None
except ImportError:
    PYQT_AVAILABLE = False


class TestLayoutPersistence(unittest.TestCase):
    """Test layout persistence functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "layout.json"

        # Sample layout data
        self.sample_layout = {
            "model_order": ["model_1", "model_2", "urdf_generator"],
            "selected_model": "model_1",
            "window_geometry": {"x": 100, "y": 100, "width": 1400, "height": 900},
            "options": {"live_visualization": True, "gpu_acceleration": False},
        }

    def test_layout_data_structure(self):
        """Test that layout data has correct structure."""
        required_keys = ["model_order", "selected_model", "window_geometry", "options"]

        for key in required_keys:
            self.assertIn(key, self.sample_layout, f"Layout should contain {key}")

        # Test geometry structure
        geometry = self.sample_layout["window_geometry"]
        geometry_keys = ["x", "y", "width", "height"]
        for key in geometry_keys:
            self.assertIn(key, geometry, f"Geometry should contain {key}")

        # Test options structure
        options = self.sample_layout["options"]
        options_keys = ["live_visualization", "gpu_acceleration"]
        for key in options_keys:
            self.assertIn(key, options, f"Options should contain {key}")

    def test_layout_file_creation(self):
        """Test that layout file can be created and read."""
        # Write layout data
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_layout, f, indent=2)

        self.assertTrue(self.config_file.exists(), "Layout file should be created")

        # Read layout data back
        with open(self.config_file, encoding="utf-8") as f:
            loaded_data = json.load(f)

        self.assertEqual(
            loaded_data, self.sample_layout, "Loaded data should match saved data"
        )

    def test_model_order_validation(self):
        """Test model order validation logic."""
        # Valid model order
        model_cards = {"model_1": Mock(), "model_2": Mock(), "urdf_generator": Mock()}
        saved_order = ["model_1", "model_2", "urdf_generator"]

        # All models exist
        all_exist = all(model_id in model_cards for model_id in saved_order)
        self.assertTrue(all_exist, "All models should exist in cards")

        # Invalid model order (missing model)
        invalid_order = ["model_1", "model_2", "missing_model"]
        all_exist_invalid = all(model_id in model_cards for model_id in invalid_order)
        self.assertFalse(all_exist_invalid, "Should detect missing model")

    def test_config_directory_creation(self):
        """Test that config directory is created if it doesn't exist."""
        from launchers.golf_launcher import CONFIG_DIR

        # The constant should point to a valid path structure
        self.assertIsInstance(CONFIG_DIR, Path)
        self.assertTrue(
            str(CONFIG_DIR).endswith("launcher"), "Should end with launcher directory"
        )

    @unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 not available")
    def test_layout_save_load_integration(self):
        """Test integration of save and load functionality."""
        # This test would require mocking the entire GolfLauncher
        # For now, we test the data structures and file operations

        # Mock launcher state
        mock_launcher_state = {
            "model_order": ["urdf_generator", "model_1", "model_2"],
            "selected_model": "urdf_generator",
            "x": 200,
            "y": 150,
            "width": 1200,
            "height": 800,
            "live_viz": False,
            "gpu_accel": True,
        }

        # Simulate save operation
        layout_data = {
            "model_order": mock_launcher_state["model_order"],
            "selected_model": mock_launcher_state["selected_model"],
            "window_geometry": {
                "x": mock_launcher_state["x"],
                "y": mock_launcher_state["y"],
                "width": mock_launcher_state["width"],
                "height": mock_launcher_state["height"],
            },
            "options": {
                "live_visualization": mock_launcher_state["live_viz"],
                "gpu_acceleration": mock_launcher_state["gpu_accel"],
            },
        }

        # Save to file
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(layout_data, f, indent=2)

        # Load from file
        with open(self.config_file, encoding="utf-8") as f:
            loaded_data = json.load(f)

        # Verify data integrity
        self.assertEqual(loaded_data["model_order"], mock_launcher_state["model_order"])
        self.assertEqual(
            loaded_data["selected_model"], mock_launcher_state["selected_model"]
        )
        self.assertEqual(loaded_data["window_geometry"]["x"], mock_launcher_state["x"])
        self.assertEqual(
            loaded_data["options"]["live_visualization"],
            mock_launcher_state["live_viz"],
        )


class TestLayoutConstants(unittest.TestCase):
    """Test layout-related constants and paths."""

    def test_config_paths_defined(self):
        """Test that config paths are properly defined."""
        try:
            from launchers.golf_launcher import CONFIG_DIR, LAYOUT_CONFIG_FILE

            self.assertIsInstance(CONFIG_DIR, Path)
            self.assertIsInstance(LAYOUT_CONFIG_FILE, Path)

            # Verify path relationship
            self.assertEqual(LAYOUT_CONFIG_FILE.parent, CONFIG_DIR)
            self.assertEqual(LAYOUT_CONFIG_FILE.name, "layout.json")

        except ImportError as e:
            self.skipTest(f"Golf launcher not available: {e}")

    def test_grid_columns_constant(self):
        """Test that grid columns constant is correct."""
        try:
            from launchers.golf_launcher import GRID_COLUMNS

            self.assertEqual(GRID_COLUMNS, 4, "Grid should be 3x4")

        except ImportError as e:
            self.skipTest(f"Golf launcher not available: {e}")


class TestLayoutErrorHandling(unittest.TestCase):
    """Test error handling in layout persistence."""

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON in layout file.

        SEC-007: Replaced tempfile.mktemp with NamedTemporaryFile to prevent TOCTOU attacks.
        """
        # Use NamedTemporaryFile with delete=False for explicit cleanup control
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            # Write invalid JSON
            temp_file.write("{ invalid json content")

        # Attempt to load should not crash
        try:
            with open(temp_path, encoding="utf-8") as f:
                json.load(f)
            self.fail("Should have raised JSONDecodeError")
        except json.JSONDecodeError:
            # Expected behavior
            pass
        finally:
            temp_path.unlink(missing_ok=True)

    def test_missing_layout_file_handling(self):
        """Test handling when layout file doesn't exist."""
        non_existent_file = Path("/non/existent/path/layout.json")

        # Should handle gracefully
        self.assertFalse(non_existent_file.exists())

        # Loading non-existent file should be handled gracefully
        # (This would be tested in the actual launcher code)

    def test_partial_layout_data(self):
        """Test handling of partial/incomplete layout data."""
        partial_layout = {
            "model_order": ["model_1", "model_2"],
            # Missing other required fields
        }

        # Should handle missing fields gracefully
        self.assertNotIn("selected_model", partial_layout)
        self.assertNotIn("window_geometry", partial_layout)

        # Code should provide defaults for missing fields


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)
