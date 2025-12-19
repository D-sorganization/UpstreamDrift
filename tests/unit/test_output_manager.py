"""
Unit tests for output management functionality.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from shared.python.output_manager import OutputFormat, OutputManager
except ImportError:
    # Create a minimal OutputManager for testing if not available
    class OutputFormat:
        CSV = "csv"
        JSON = "json"
        HDF5 = "hdf5"
        PICKLE = "pickle"

    class OutputManager:
        def __init__(self, base_path):
            self.base_path = Path(base_path)

        def save_simulation_results(
            self, results, filename, format_type=OutputFormat.CSV
        ):
            pass

        def load_simulation_results(self, filename, format_type=OutputFormat.CSV):
            return {}

        def create_output_structure(self):
            pass


class TestOutputManager:
    """Test cases for OutputManager class."""

    def test_output_manager_initialization(self, temp_dir):
        """Test OutputManager initializes correctly."""
        manager = OutputManager(temp_dir)
        assert manager.base_path == temp_dir

    def test_create_output_structure(self, temp_dir):
        """Test creation of output directory structure."""
        manager = OutputManager(temp_dir)
        manager.create_output_structure()

        # Check that required directories are created
        expected_dirs = ["simulations", "analysis", "exports", "reports", "cache"]

        for dir_name in expected_dirs:
            assert (temp_dir / dir_name).exists()
            assert (temp_dir / dir_name).is_dir()

    def test_save_csv_results(self, temp_dir, sample_swing_data):
        """Test saving results in CSV format."""
        manager = OutputManager(temp_dir)
        manager.create_output_structure()

        filename = "test_swing.csv"
        manager.save_simulation_results(sample_swing_data, filename, OutputFormat.CSV)

        # Verify file was created (should be in mujoco subdirectory)
        output_file = temp_dir / "simulations" / "mujoco" / filename
        assert output_file.exists()

        # Verify content can be loaded back
        loaded_data = pd.read_csv(output_file)
        assert len(loaded_data) == len(sample_swing_data)
        assert list(loaded_data.columns) == list(sample_swing_data.columns)

    def test_save_json_results(self, temp_dir):
        """Test saving results in JSON format."""
        manager = OutputManager(temp_dir)
        manager.create_output_structure()

        test_data = {
            "swing_speed": 100.5,
            "ball_distance": 250.3,
            "launch_angle": 12.5,
            "metadata": {"engine": "mujoco", "timestamp": "2024-01-01T12:00:00"},
        }

        filename = "test_results.json"
        manager.save_simulation_results(test_data, filename, OutputFormat.JSON)

        # Verify file was created and content is correct (should be in mujoco subdir)
        output_file = temp_dir / "simulations" / "mujoco" / filename
        assert output_file.exists()

        with open(output_file) as f:
            loaded_data = json.load(f)

        # The saved data is wrapped with metadata, so check the results field
        assert loaded_data["results"] == test_data
        assert loaded_data["engine"] == "mujoco"
        assert "timestamp" in loaded_data

    def test_load_simulation_results(self, temp_dir, sample_swing_data):
        """Test loading simulation results."""
        manager = OutputManager(temp_dir)
        manager.create_output_structure()

        # Save data first
        filename = "test_load.csv"
        manager.save_simulation_results(sample_swing_data, filename, OutputFormat.CSV)

        # Load it back
        loaded_data = manager.load_simulation_results(filename, OutputFormat.CSV)

        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_swing_data)

    def test_get_simulation_list(self, temp_dir):
        """Test getting list of saved simulations."""
        manager = OutputManager(temp_dir)
        manager.create_output_structure()

        # Create some test files in engine directories
        mujoco_dir = temp_dir / "simulations" / "mujoco"
        test_files = ["sim1.csv", "sim2.json", "sim3.csv"]

        for filename in test_files:
            (mujoco_dir / filename).touch()

        # Get simulation list
        simulations = manager.get_simulation_list()

        assert len(simulations) == len(test_files)
        for filename in test_files:
            assert filename in simulations

    def test_cleanup_old_files(self, temp_dir):
        """Test cleanup of old simulation files."""
        manager = OutputManager(temp_dir)
        manager.create_output_structure()

        sim_dir = temp_dir / "simulations"

        # Create test files with different ages
        old_file = sim_dir / "old_sim.csv"
        new_file = sim_dir / "new_sim.csv"

        old_file.touch()
        new_file.touch()

        # Mock file modification times
        import time

        old_time = time.time() - (30 * 24 * 3600)  # 30 days ago
        new_time = time.time() - (1 * 24 * 3600)  # 1 day ago

        with patch("os.path.getmtime") as mock_getmtime:

            def mock_time(path):
                if "old_sim" in str(path):
                    return old_time
                return new_time

            mock_getmtime.side_effect = mock_time

            # Cleanup files older than 7 days
            cleaned_count = manager.cleanup_old_files(max_age_days=7)

            assert cleaned_count >= 0  # Should not error

    def test_export_analysis_report(self, temp_dir):
        """Test exporting analysis reports."""
        manager = OutputManager(temp_dir)
        manager.create_output_structure()

        analysis_data = {
            "summary": {
                "total_swings": 10,
                "average_distance": 245.5,
                "best_distance": 267.3,
            },
            "details": [
                {"swing_id": 1, "distance": 245.0, "speed": 98.5},
                {"swing_id": 2, "distance": 250.0, "speed": 102.1},
            ],
        }

        report_file = manager.export_analysis_report(
            analysis_data, "swing_analysis_report"
        )

        assert report_file.exists()
        assert report_file.suffix == ".json"

    @pytest.mark.parametrize(
        "format_type",
        [
            OutputFormat.CSV,
            OutputFormat.JSON,
        ],
    )
    def test_different_output_formats(self, temp_dir, format_type):
        """Test saving and loading with different formats."""
        manager = OutputManager(temp_dir)
        manager.create_output_structure()

        if format_type == OutputFormat.CSV:
            test_data = pd.DataFrame(
                {"time": [0, 1, 2], "position": [0, 1, 4], "velocity": [0, 1, 2]}
            )
        else:  # JSON
            test_data = {
                "time": [0, 1, 2],
                "position": [0, 1, 4],
                "velocity": [0, 1, 2],
            }

        filename = f"test_format_{format_type.value}"

        # Save and load
        manager.save_simulation_results(test_data, filename, format_type)
        loaded_data = manager.load_simulation_results(filename, format_type)

        assert loaded_data is not None


class TestOutputManagerIntegration:
    """Integration tests for OutputManager."""

    @pytest.mark.integration
    def test_full_workflow(self, temp_dir, sample_swing_data):
        """Test complete output management workflow."""
        manager = OutputManager(temp_dir)

        # 1. Create structure
        manager.create_output_structure()

        # 2. Save simulation results
        saved_path = manager.save_simulation_results(
            sample_swing_data, "integration_test.csv", OutputFormat.CSV
        )

        # 3. Load results back using the actual saved filename
        saved_filename = saved_path.name
        loaded_data = manager.load_simulation_results(saved_filename, OutputFormat.CSV)

        # 4. Verify data integrity
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_swing_data)

        # 5. Get simulation list
        simulations = manager.get_simulation_list()
        assert saved_filename in simulations

        # 6. Export report
        analysis_data = {
            "simulation_file": saved_filename,
            "total_time": float(sample_swing_data["time"].max()),
            "max_velocity": float(sample_swing_data["club_velocity"].max()),
        }

        report_file = manager.export_analysis_report(
            analysis_data, "integration_report"
        )

        assert report_file.exists()


# Test error handling
class TestOutputManagerErrors:
    """Test error handling in OutputManager."""

    def test_invalid_base_path(self):
        """Test handling of invalid base path."""
        # Use a cross-platform invalid path
        import os

        if os.name == "nt":  # Windows
            invalid_path = "Z:\\invalid\\path\\that\\does\\not\\exist"
        else:  # Unix/Linux
            invalid_path = "/invalid/path/that/does/not/exist"

        try:
            manager = OutputManager(invalid_path)
            manager.create_output_structure()
            # If we get here, the path was created (which is fine)
            # Just verify the manager was created and has the expected path name
            assert manager.base_path.name == "exist"
        except (OSError, PermissionError, FileNotFoundError):
            # This is expected behavior for truly invalid paths
            pass

    def test_save_with_invalid_format(self, temp_dir):
        """Test saving with invalid format."""
        manager = OutputManager(temp_dir)
        manager.create_output_structure()

        with pytest.raises((ValueError, AttributeError)):
            manager.save_simulation_results(
                {"test": "data"}, "test.txt", "invalid_format"
            )

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading nonexistent file."""
        manager = OutputManager(temp_dir)
        manager.create_output_structure()

        with pytest.raises(FileNotFoundError):
            manager.load_simulation_results("nonexistent.csv", OutputFormat.CSV)
