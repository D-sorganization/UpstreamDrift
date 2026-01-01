"""Unit tests for shared output manager."""

import shutil
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from shared.python.output_manager import OutputFormat, OutputManager


class TestOutputManager(unittest.TestCase):
    """Test cases for OutputManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path.cwd() / "tests" / "output_test_temp"
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.manager = OutputManager(self.test_dir)
        self.manager.create_output_structure()

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
            except (PermissionError, OSError):
                pass

    def test_initialization_directory_creation(self):
        """Test that directory structure is created."""
        self.manager.create_output_structure()

        self.assertTrue((self.test_dir / "simulations" / "mujoco").exists())
        self.assertTrue((self.test_dir / "analysis" / "biomechanics").exists())
        self.assertTrue((self.test_dir / "reports").exists())

    def test_save_load_csv(self):
        """Test saving and loading CSV files."""
        data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        df = pd.DataFrame(data)

        path = self.manager.save_simulation_results(
            df, "test_csv", format_type=OutputFormat.CSV
        )

        self.assertTrue(path.exists())
        loaded_df = self.manager.load_simulation_results(
            path.stem, format_type=OutputFormat.CSV
        )
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_save_load_json(self):
        """Test saving and loading JSON files."""
        data = {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}}

        path = self.manager.save_simulation_results(
            data, "test_json", format_type=OutputFormat.JSON
        )

        self.assertTrue(path.exists())
        loaded_data = self.manager.load_simulation_results(
            path.stem, format_type=OutputFormat.JSON
        )
        self.assertEqual(loaded_data, data)

    def test_save_load_pickle(self):
        """Test that Pickle format raises security error."""
        data = {"key": "value", "array": np.array([1, 2, 3])}

        with self.assertRaisesRegex(ValueError, "Pickle format is disabled"):
            self.manager.save_simulation_results(
                data, "test_pickle", format_type=OutputFormat.PICKLE
            )

    def test_save_json_numpy_serialization(self):
        """Test JSON serialization of NumPy types."""
        data = {
            "array": np.array([1, 2, 3]),
            "float": np.float64(1.5),
            "int": np.int64(10),
        }

        path = self.manager.save_simulation_results(
            data, "test_numpy_json", format_type=OutputFormat.JSON
        )

        loaded_data = self.manager.load_simulation_results(
            path.stem, format_type=OutputFormat.JSON
        )

        self.assertEqual(loaded_data["array"], [1, 2, 3])
        self.assertEqual(loaded_data["float"], 1.5)
        self.assertEqual(loaded_data["int"], 10)

    def test_get_simulation_list(self):
        """Test retrieving list of simulations."""
        # Manually create files to verify list logic independent of save logic
        sim_dir = self.manager.directories["simulations"] / "mujoco"
        sim_dir.mkdir(parents=True, exist_ok=True)

        (sim_dir / "sim1.csv").touch()
        (sim_dir / "sim2.json").touch()

        sims = self.manager.get_simulation_list(engine="mujoco")

        self.assertIn("sim1.csv", sims)
        self.assertIn("sim2.json", sims)

    def test_export_analysis_report_html(self):
        """Test exporting HTML report."""
        data = {"summary": "test", "value": 123}
        path = self.manager.export_analysis_report(
            data, "test_report", format_type="html"
        )

        self.assertTrue(path.exists())
        with open(path) as f:
            content = f.read()
            self.assertIn("<html>", content)
            self.assertIn("test", content)
            self.assertIn("123", content)

    def test_cleanup_old_files(self):
        """Test cleaning up old files."""
        self.manager.create_output_structure()

        # We use strict mocking to avoid filesystem timestamp issues and OS errors with mocks
        with patch("shared.python.output_manager.datetime") as mock_datetime:
            # Fix "now" to a future time to ensure created files appear old
            fixed_now = datetime(2099, 1, 10, 12, 0, 0)
            mock_datetime.now.return_value = fixed_now
            mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp

            # Create a file that we will pretend is old
            old_file = self.manager.directories["cache"] / "temp" / "old.txt"
            old_file.parent.mkdir(parents=True, exist_ok=True)
            old_file.touch()

            # Also mock unlink to verify it was called
            with patch.object(Path, "unlink") as mock_unlink:
                cleaned = self.manager.cleanup_old_files()

                self.assertGreaterEqual(cleaned, 1)
                self.assertTrue(mock_unlink.called)
