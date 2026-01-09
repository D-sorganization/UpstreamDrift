from pathlib import Path

import matplotlib.pyplot as plt
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


def test_ensure_output_dir(tmp_path):
    # Mock OUTPUT_ROOT by patching it or just testing relative paths if possible
    # But ensure_output_dir imports OUTPUT_ROOT from . which resolves to shared/python/__init__.py
    # We can rely on it creating a directory.
    # However, it uses a global constant.
    # Let's check if it creates the directory.

    # Since we can't easily mock the internal import without more complex mocking,
    # we'll trust it works if we don't crash, but better to verify side effects.
    # Actually, ensure_output_dir creates directories under OUTPUT_ROOT.
    # If OUTPUT_ROOT is absolute, we might pollute.
    # But we can assume it's relative or safe in test env.

    # Mocking shared.python.OUTPUT_ROOT
    import shared.python

    # Check if OUTPUT_ROOT exists in shared.python
    if hasattr(shared.python, "OUTPUT_ROOT"):
        original_root = shared.python.OUTPUT_ROOT
        shared.python.OUTPUT_ROOT = tmp_path
        try:
            path = ensure_output_dir("test_engine", "subdir")
            assert path == tmp_path / "test_engine" / "subdir"
            assert path.exists()
        finally:
            shared.python.OUTPUT_ROOT = original_root
    else:
        # Fallback if mocking fails, just check if it returns a path
        path = ensure_output_dir("test_engine", "subdir")
        assert isinstance(path, Path)
        assert path.exists()


def test_save_and_load_golf_data(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    # CSV
    csv_path = tmp_path / "test.csv"
    save_golf_data(df, csv_path, "csv")
    loaded_df = load_golf_data(csv_path)
    pd.testing.assert_frame_equal(df, loaded_df)

    # JSON
    json_path = tmp_path / "test.json"
    save_golf_data(df, json_path, "json")
    loaded_df_json = load_golf_data(json_path)
    pd.testing.assert_frame_equal(df, loaded_df_json)

    # Excel
    # Requires openpyxl which might not be installed, skip if fails
    try:
        xlsx_path = tmp_path / "test.xlsx"
        save_golf_data(df, xlsx_path, "excel")
        loaded_df_xlsx = load_golf_data(xlsx_path)
        pd.testing.assert_frame_equal(df, loaded_df_xlsx)
    except ImportError:
        pass

    # Unsupported format save
    with pytest.raises(ValueError):
        save_golf_data(df, tmp_path / "test.txt", "txt")

    # Unsupported format load
    with pytest.raises(ValueError):
        load_golf_data(tmp_path / "test.txt")


def test_standardize_joint_angles():
    angles = np.array([[0.1, 0.2], [0.3, 0.4]])
    df = standardize_joint_angles(angles, ["j1", "j2"])
    assert "time" in df.columns
    assert "j1" in df.columns
    assert df.iloc[0]["j1"] == 0.1

    # Default names
    df_default = standardize_joint_angles(angles)
    assert "joint_0" in df_default.columns


def test_plot_joint_trajectories(tmp_path):
    df = pd.DataFrame(
        {
            "time": [0, 1],
            "joint1": [0, 1],
            "joint2": [1, 0],
            "joint3": [0, 0],
            "joint4": [1, 1],
            "joint5": [0, 1],  # Should be ignored in top 4
        }
    )

    save_path = tmp_path / "plot.png"
    fig = plot_joint_trajectories(df, title="Test Plot", save_path=save_path)

    assert isinstance(fig, plt.Figure)
    assert save_path.exists()
    plt.close(fig)


def test_convert_units():
    assert convert_units(180, "deg", "rad") == pytest.approx(np.pi)
    assert convert_units(np.pi, "rad", "deg") == pytest.approx(180)
    assert convert_units(1, "m", "mm") == 1000
    assert convert_units(1000, "mm", "m") == 1
    assert convert_units(1, "m/s", "mph") == pytest.approx(2.237, 0.001)
    assert convert_units(2.237, "mph", "m/s") == pytest.approx(1, 0.001)

    with pytest.raises(ValueError):
        convert_units(1, "kg", "lb")


def test_get_shared_urdf_path():
    # This depends on the environment, but we can check it returns a Path or None
    path = get_shared_urdf_path()
    if path is not None:
        assert isinstance(path, Path)
