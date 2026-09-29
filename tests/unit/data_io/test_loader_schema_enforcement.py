"""End-to-end schema-enforcement tests for data_io loaders (issue #6568).

Confirms valid files still load identically while malformed files surface a
typed error (directly, or as the documented soft-fail ``None`` for loaders
wrapped by ``@log_errors(reraise=False)``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.shared.python.data_io.common_utils import load_golf_data
from src.shared.python.data_io.data_utils import load_csv_data
from src.shared.python.data_io.swing_capture_import import SwingCaptureImporter

pytestmark = pytest.mark.unit


def test_load_csv_data_valid(tmp_path: Path) -> None:
    csv_path = tmp_path / "good.csv"
    pd.DataFrame({"time": [0.0, 0.1], "q0": [1.0, 2.0]}).to_csv(csv_path, index=False)
    df = load_csv_data(csv_path)
    assert df is not None
    assert list(df.columns) == ["time", "q0"]
    assert len(df) == 2


def test_load_csv_data_empty_soft_fails(tmp_path: Path) -> None:
    # @log_errors(reraise=False, default_return=None): malformed input now
    # returns None (logged DataFormatError) instead of a columnless frame.
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")
    assert load_csv_data(csv_path) is None


def test_load_golf_data_valid(tmp_path: Path) -> None:
    csv_path = tmp_path / "golf.csv"
    pd.DataFrame({"time": [0.0, 0.1], "q0": [1.0, 2.0]}).to_csv(csv_path, index=False)
    df = load_golf_data(csv_path)
    assert list(df.columns) == ["time", "q0"]


def test_swing_capture_import_csv_valid(tmp_path: Path) -> None:
    csv_path = tmp_path / "traj.csv"
    csv_path.write_text(
        "time,joint_0,joint_1\n0.0,0.1,0.2\n0.1,0.3,0.4\n0.2,0.5,0.6\n",
        encoding="utf-8",
    )
    traj = SwingCaptureImporter().import_csv(csv_path)
    assert traj.joint_names == ["joint_0", "joint_1"]
    assert traj.positions.shape == (3, 2)


def test_swing_capture_import_csv_single_column_raises(tmp_path: Path) -> None:
    # A single data cell -> array reshapes to (1, 1) -> too few columns
    # (no joint data) -> typed DataFormatError instead of an empty trajectory.
    from src.shared.python.core.error_utils import DataFormatError

    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("time\n0.0\n", encoding="utf-8")
    with pytest.raises(DataFormatError, match="too few columns|must be 2D"):
        SwingCaptureImporter().import_csv(csv_path)


def test_swing_capture_import_csv_nan_raises(tmp_path: Path) -> None:
    from src.shared.python.core.error_utils import DataFormatError

    csv_path = tmp_path / "nan.csv"
    csv_path.write_text("time,j0\n0.0,1.0\n0.1,nan\n", encoding="utf-8")
    with pytest.raises(DataFormatError, match="non-finite"):
        SwingCaptureImporter().import_csv(csv_path)


def test_numpy_loaded_trajectory_array_shape() -> None:
    # Sanity: validate_trajectory_array accepts a typical loadtxt output.
    from src.shared.python.data_io._schemas import validate_trajectory_array

    arr = np.array([[0.0, 1.0], [0.1, 2.0]])
    assert validate_trajectory_array(arr, source="mem") is arr
