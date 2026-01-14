from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from shared.python.common_utils import (
    convert_units,
    ensure_output_dir,
    get_shared_urdf_path,
    load_golf_data,
    plot_joint_trajectories,
    save_golf_data,
    standardize_joint_angles,
)


def test_convert_units():
    """Test unit conversion utility."""
    assert convert_units(180, "deg", "rad") == pytest.approx(np.pi)
    assert convert_units(np.pi, "rad", "deg") == pytest.approx(180)
    assert convert_units(1, "m", "mm") == 1000
    assert convert_units(1000, "mm", "m") == 1
    assert convert_units(10, "m/s", "mph") == pytest.approx(22.3694, rel=1e-4)
    assert convert_units(22.3694, "mph", "m/s") == pytest.approx(10, rel=1e-4)

    with pytest.raises(ValueError, match="not supported"):
        convert_units(1, "furlongs", "fortnights")


def test_ensure_output_dir(tmp_path):
    """Test output directory creation."""
    with patch("shared.python.OUTPUT_ROOT", tmp_path / "outputs"):
        path = ensure_output_dir("test_engine", "run1")
        assert path.exists()
        assert path.name == "run1"
        assert path.parent.name == "test_engine"


def test_load_save_golf_data(tmp_path):
    """Test loading and saving golf data."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    # CSV
    csv_path = tmp_path / "test.csv"
    save_golf_data(df, csv_path, "csv")
    loaded_df = load_golf_data(csv_path)
    pd.testing.assert_frame_equal(df, loaded_df)

    # Excel - requires openpyxl, handle if missing
    try:
        import importlib.util

        if importlib.util.find_spec("openpyxl") is not None:
            xlsx_path = tmp_path / "test.xlsx"
            save_golf_data(df, xlsx_path, "excel")
            loaded_df_xlsx = load_golf_data(xlsx_path)
            pd.testing.assert_frame_equal(df, loaded_df_xlsx)
    except ImportError:
        pass

    # JSON
    json_path = tmp_path / "test.json"
    save_golf_data(df, json_path, "json")
    loaded_df_json = load_golf_data(json_path)
    pd.testing.assert_frame_equal(df, loaded_df_json)

    # Invalid format
    with pytest.raises(ValueError):
        save_golf_data(df, tmp_path / "test.txt", "txt")

    with pytest.raises(ValueError):
        load_golf_data(tmp_path / "test.txt")


def test_standardize_joint_angles():
    """Test joint angle standardization."""
    angles = np.array([[0, 1], [0.1, 1.1]])
    df = standardize_joint_angles(angles)
    assert "joint_0" in df.columns
    assert "joint_1" in df.columns
    assert "time" in df.columns
    assert len(df) == 2


def test_plot_joint_trajectories():
    """Test plotting function."""
    df = pd.DataFrame({"time": [0, 1], "joint_0": [0, 1], "joint_1": [2, 3]})

    # Only test that it runs without error (mocking plt to avoid display)
    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.subplots") as mock_subplots,
        patch("matplotlib.pyplot.savefig"),
    ):
        mock_fig = MagicMock()
        # Explicitly create and assign to object array to ensure no None values
        mock_arr = np.empty((2, 2), dtype=object)
        m1, m2, m3, m4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        mock_arr[0, 0] = m1
        mock_arr[0, 1] = m2
        mock_arr[1, 0] = m3
        mock_arr[1, 1] = m4
        mock_subplots.return_value = (mock_fig, mock_arr)

        fig = plot_joint_trajectories(df, title="Test Plot")
        assert fig is mock_fig


def test_get_shared_urdf_path():
    """Test URDF path resolution."""
    # This relies on the actual filesystem of the repo, might return None or Path
    path = get_shared_urdf_path()
    # Just verify it returns None or a valid Path, covering the code
    assert path is None or isinstance(path, Path)
