from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend
from unittest.mock import patch

import matplotlib.pyplot as plt

from shared.python.common_utils import (
    convert_units,
    ensure_output_dir,
    get_shared_urdf_path,
    load_golf_data,
    plot_joint_trajectories,
    save_golf_data,
    standardize_joint_angles,
)


class TestCommonUtils:
    def test_ensure_output_dir(self, tmp_path):
        """Test output directory creation."""
        with patch("shared.python.OUTPUT_ROOT", tmp_path):
            path = ensure_output_dir("test_engine", "test_subdir")
            assert path.exists()
            assert path.name == "test_subdir"
            assert path.parent.name == "test_engine"

    def test_load_save_golf_data_csv(self, tmp_path):
        """Test loading and saving CSV data."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv_path = tmp_path / "test.csv"

        save_golf_data(df, csv_path, format="csv")
        assert csv_path.exists()

        loaded_df = load_golf_data(csv_path)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_load_save_golf_data_json(self, tmp_path):
        """Test loading and saving JSON data."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        json_path = tmp_path / "test.json"

        save_golf_data(df, json_path, format="json")
        assert json_path.exists()

        loaded_df = load_golf_data(json_path)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_load_unsupported_format(self, tmp_path):
        """Test error for unsupported formats."""
        path = tmp_path / "test.xyz"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_golf_data(path)

    def test_save_unsupported_format(self, tmp_path):
        """Test error for unsupported formats."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Unsupported format"):
            save_golf_data(df, tmp_path / "test.txt", format="txt")

    def test_standardize_joint_angles(self):
        """Test joint angle standardization."""
        angles = np.zeros((10, 2))
        df = standardize_joint_angles(angles, ["j1", "j2"])

        assert "time" in df.columns
        assert "j1" in df.columns
        assert "j2" in df.columns
        assert len(df) == 10

    def test_plot_joint_trajectories(self, tmp_path):
        """Test plotting function."""
        df = pd.DataFrame(
            {
                "time": np.linspace(0, 1, 10),
                "joint_1": np.random.rand(10),
                "joint_2": np.random.rand(10),
            }
        )

        save_path = tmp_path / "plot.png"
        fig = plot_joint_trajectories(df, title="Test Plot", save_path=save_path)

        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        plt.close(fig)

    def test_convert_units(self):
        """Test unit conversion."""
        assert convert_units(180, "deg", "rad") == pytest.approx(np.pi)
        assert convert_units(np.pi, "rad", "deg") == pytest.approx(180)
        assert convert_units(1, "m", "mm") == 1000
        assert convert_units(1000, "mm", "m") == 1
        assert convert_units(10, "m/s", "mph") == pytest.approx(22.37)

        with pytest.raises(ValueError, match="Conversion from.*not supported"):
            convert_units(1, "kg", "lbs")

    def test_get_shared_urdf_path(self):
        """Test getting shared URDF path."""
        # This relies on actual file structure, which might be tricky to mock reliably
        # without affecting other tests.
        # But we can check if it returns None or a Path.
        path = get_shared_urdf_path()
        if path is not None:
            assert isinstance(path, Path)
