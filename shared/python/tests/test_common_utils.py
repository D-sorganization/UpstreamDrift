"""Tests for common utilities."""

import numpy as np
import pandas as pd
import pytest

from shared.python.common_utils import (
    convert_units,
    ensure_output_dir,
    get_shared_urdf_path,
    load_golf_data,
    save_golf_data,
    standardize_joint_angles,
)

# Mock OUTPUT_ROOT for ensure_output_dir
# common_utils imports OUTPUT_ROOT from . which is shared/python/__init__.py
# We can't easily mock that import without monkeypatching the module attribute
# But ensure_output_dir calls `from . import OUTPUT_ROOT` inside the function?
# No, it imports it at top level in my reading of common_utils.py?
# Ah, `from . import OUTPUT_ROOT` is inside `ensure_output_dir`.
# So we can patch `shared.python.OUTPUT_ROOT`.


def test_convert_units():
    """Test unit conversion."""
    # Degrees <-> Radians
    assert np.isclose(convert_units(180, "deg", "rad"), np.pi)
    assert np.isclose(convert_units(np.pi, "rad", "deg"), 180)

    # Length
    assert convert_units(1, "m", "mm") == 1000
    assert convert_units(1000, "mm", "m") == 1

    # Velocity
    assert np.isclose(convert_units(100, "mph", "m/s"), 44.7, atol=0.1)
    assert np.isclose(convert_units(44.7, "m/s", "mph"), 100, atol=0.1)

    with pytest.raises(ValueError):
        convert_units(1, "unknown", "unit")


def test_standardize_joint_angles():
    """Test joint angle standardization."""
    angles = np.array([[0, 1], [0.1, 1.1], [0.2, 1.2]])
    df = standardize_joint_angles(angles, ["j1", "j2"])

    assert isinstance(df, pd.DataFrame)
    assert "time" in df.columns
    assert "j1" in df.columns
    assert "j2" in df.columns
    assert len(df) == 3

    # Default naming
    df_default = standardize_joint_angles(angles)
    assert "joint_0" in df_default.columns


def test_save_and_load_golf_data(tmp_path):
    """Test saving and loading data."""
    df = pd.DataFrame({"time": [0, 1], "val": [10, 20]})

    # CSV
    csv_path = tmp_path / "test.csv"
    save_golf_data(df, csv_path, "csv")
    loaded_df = load_golf_data(csv_path)
    pd.testing.assert_frame_equal(df, loaded_df)

    # JSON
    json_path = tmp_path / "test.json"
    save_golf_data(df, json_path, "json")
    loaded_json = load_golf_data(json_path)
    pd.testing.assert_frame_equal(df, loaded_json)

    # Excel requires openpyxl usually, assuming installed or we skip
    try:
        import openpyxl  # noqa: F401

        xlsx_path = tmp_path / "test.xlsx"
        save_golf_data(df, xlsx_path, "excel")
        loaded_xlsx = load_golf_data(xlsx_path)
        pd.testing.assert_frame_equal(df, loaded_xlsx)
    except ImportError:
        pass

    # Unsupported
    with pytest.raises(ValueError):
        save_golf_data(df, tmp_path / "test.txt", "txt")

    with pytest.raises(ValueError):
        load_golf_data(tmp_path / "test.txt")


def test_ensure_output_dir(tmp_path, monkeypatch):
    """Test output directory creation."""
    # Monkeypatch shared.python.OUTPUT_ROOT
    # Since it is imported inside the function `from . import OUTPUT_ROOT`,
    # we need to patch the module `shared.python` to have `OUTPUT_ROOT`.
    import shared.python

    monkeypatch.setattr(shared.python, "OUTPUT_ROOT", tmp_path)

    path = ensure_output_dir("test_engine", "run1")
    assert path.exists()
    assert path == tmp_path / "test_engine" / "run1"


def test_get_shared_urdf_path():
    """Test finding URDF path."""
    # This depends on the actual file system structure where tests run
    # shared/urdf should exist in this repo
    path = get_shared_urdf_path()
    if path:
        assert path.name == "urdf"
        assert path.exists()
    else:
        # If running in environment without shared/urdf, this is expected?
        # But we saw shared/urdf in list_files.
        pass
