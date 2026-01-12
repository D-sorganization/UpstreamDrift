from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from shared.python.common_utils import (
    DataFormatError,
    convert_units,
    ensure_output_dir,
    get_shared_urdf_path,
    load_golf_data,
    plot_joint_trajectories,
    save_golf_data,
    standardize_joint_angles,
)
from shared.python.constants import MPS_TO_MPH

# Use non-interactive backend for plots
matplotlib.use("Agg")


class TestCommonUtils:

    @pytest.fixture
    def mock_output_root(self, tmp_path):
        """Mock OUTPUT_ROOT in shared.python"""
        with patch("shared.python.OUTPUT_ROOT", tmp_path):
            yield tmp_path

    def test_ensure_output_dir(self, mock_output_root):
        engine_name = "test_engine"
        path = ensure_output_dir(engine_name)

        assert path.exists()
        assert path == mock_output_root / engine_name

        subdir_path = ensure_output_dir(engine_name, subdir="run_1")
        assert subdir_path.exists()
        assert subdir_path == mock_output_root / engine_name / "run_1"

    def test_load_golf_data_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        loaded_df = load_golf_data(csv_path)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_load_golf_data_json(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        json_path = tmp_path / "test.json"
        df.to_json(json_path, orient="records", indent=2)

        loaded_df = load_golf_data(json_path)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_load_golf_data_unsupported(self, tmp_path):
        txt_path = tmp_path / "test.txt"
        txt_path.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_golf_data(txt_path)

    def test_save_golf_data(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Test CSV
        csv_path = tmp_path / "out.csv"
        save_golf_data(df, csv_path, format="csv")
        assert csv_path.exists()
        pd.testing.assert_frame_equal(pd.read_csv(csv_path), df)

        # Test JSON
        json_path = tmp_path / "out.json"
        save_golf_data(df, json_path, format="json")
        assert json_path.exists()
        pd.testing.assert_frame_equal(pd.read_json(json_path), df)

        # Test Invalid
        with pytest.raises(ValueError, match="Unsupported format"):
            save_golf_data(df, tmp_path / "out.xyz", format="xyz")

    def test_standardize_joint_angles(self):
        angles = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
        df = standardize_joint_angles(angles, time_step=0.1)

        assert "time" in df.columns
        assert list(df.columns) == ["joint_0", "joint_1", "time"]
        assert len(df) == 3
        # Check time vector
        expected_time = np.array([0.0, 0.1, 0.2])
        np.testing.assert_allclose(df["time"].to_numpy(), expected_time)

    def test_standardize_joint_angles_custom_names(self):
        angles = np.array([[0.1], [0.2]])
        df = standardize_joint_angles(angles, angle_names=["hip"], time_step=1.0)
        assert list(df.columns) == ["hip", "time"]

    def test_plot_joint_trajectories(self, tmp_path):
        df = pd.DataFrame({"time": [0, 1, 2], "joint1": [0, 1, 2], "joint2": [2, 1, 0]})

        fig = plot_joint_trajectories(df, title="Test Plot")
        assert isinstance(fig, plt.Figure)

        # Test saving
        save_path = tmp_path / "plot.png"
        plot_joint_trajectories(df, save_path=save_path)
        assert save_path.exists()

        plt.close(fig)

    def test_convert_units(self):
        assert np.isclose(convert_units(180, "deg", "rad"), np.pi)
        assert np.isclose(convert_units(np.pi, "rad", "deg"), 180.0)

        assert convert_units(1.0, "m", "mm") == 1000.0
        assert convert_units(1000.0, "mm", "m") == 1.0

        # 1 m/s approx 2.23694 mph
        assert np.isclose(convert_units(1.0, "m/s", "mph"), float(MPS_TO_MPH))

        with pytest.raises(ValueError):
            convert_units(1.0, "kg", "lbs")

    def test_get_shared_urdf_path(self):
        # This test relies on the actual repo structure or needs complex mocking.
        # We'll just verify it returns a Path or None
        path = get_shared_urdf_path()
        if path:
            assert isinstance(path, Path)

    def test_exceptions_import(self):
        # Just verify we can instantiate them
        err = DataFormatError("test")
        assert str(err) == "test"
