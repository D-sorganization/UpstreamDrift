"""Unit tests for shared output manager."""

import shutil
import unittest
from datetime import datetime
from pathlib import Path

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
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.manager = OutputManager(self.test_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

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
        """Test saving and loading Pickle files."""
        data = {"key": "value", "array": np.array([1, 2, 3])}

        path = self.manager.save_simulation_results(
            data, "test_pickle", format_type=OutputFormat.PICKLE
        )

        self.assertTrue(path.exists())
        loaded_data = self.manager.load_simulation_results(
            path.stem, format_type=OutputFormat.PICKLE
        )

        self.assertEqual(loaded_data["key"], data["key"])
        np.testing.assert_array_equal(loaded_data["array"], data["array"])

    def test_save_json_numpy_serialization(self):
        """Test JSON serialization of NumPy types."""
        data = {
            "array": np.array([1, 2, 3]),
            "float": np.float64(1.5),
            "int": np.int64(10)
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

        # Create a dummy old file
        old_file = self.manager.directories["cache"] / "temp" / "old.txt"
        old_file.parent.mkdir(parents=True, exist_ok=True)
        old_file.touch()

        # Modify mtime to be old
        old_time = (datetime.now() - pd.Timedelta(days=2)).timestamp()
        import os
        os.utime(old_file, (old_time, old_time))

        # Create a new file
        new_file = self.manager.directories["cache"] / "temp" / "new.txt"
        new_file.touch()

        cleaned = self.manager.cleanup_old_files()

        self.assertEqual(cleaned, 1)
        self.assertFalse(old_file.exists())
        self.assertTrue(new_file.exists())
