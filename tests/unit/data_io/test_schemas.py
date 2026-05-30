"""Tests for additive pandera schema validation in data_io (issue #6568).

Validation must be additive: valid input behaves exactly as before, while
malformed input raises a typed :class:`DataFormatError`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.shared.python.core.error_utils import DataFormatError
from src.shared.python.data_io._schemas import (
    pandera_available,
    validate_csv_dataframe,
    validate_trajectory_array,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# validate_csv_dataframe
# ---------------------------------------------------------------------------


def test_valid_dataframe_passes_through_unchanged() -> None:
    df = pd.DataFrame({"time": [0.0, 0.1], "q0": [1.0, 2.0]})
    result = validate_csv_dataframe(df, source="ok.csv")
    assert result is df  # identity preserved on success path


def test_empty_columns_raises() -> None:
    df = pd.DataFrame(index=[0, 1])  # rows but zero columns
    with pytest.raises(DataFormatError, match="zero columns"):
        validate_csv_dataframe(df, source="bad.csv")


def test_missing_required_column_raises() -> None:
    df = pd.DataFrame({"q0": [1.0]})
    with pytest.raises(DataFormatError, match="missing required column"):
        validate_csv_dataframe(df, source="bad.csv", required_columns=("time", "q0"))


def test_required_columns_present_passes() -> None:
    df = pd.DataFrame({"time": [0.0], "q0": [1.0]})
    result = validate_csv_dataframe(
        df, source="ok.csv", required_columns=("time", "q0")
    )
    assert result is df


def test_require_numeric_rejects_string_columns() -> None:
    df = pd.DataFrame({"time": [0.0, 0.1], "label": ["a", "b"]})
    with pytest.raises(DataFormatError):
        validate_csv_dataframe(df, source="bad.csv", require_numeric=True)


def test_require_numeric_accepts_all_numeric() -> None:
    df = pd.DataFrame({"time": [0.0, 0.1], "q0": [1.0, 2.0]})
    result = validate_csv_dataframe(df, source="ok.csv", require_numeric=True)
    assert result is df


def test_none_dataframe_raises_value_error() -> None:
    with pytest.raises(ValueError, match="data must be provided"):
        validate_csv_dataframe(None, source="x.csv")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_trajectory_array
# ---------------------------------------------------------------------------


def test_valid_trajectory_passes_through_unchanged() -> None:
    arr = np.array([[0.0, 1.0, 2.0], [0.1, 1.5, 2.5]])
    result = validate_trajectory_array(arr, source="ok.csv")
    assert result is arr


def test_trajectory_must_be_2d() -> None:
    arr = np.array([0.0, 1.0, 2.0])
    with pytest.raises(DataFormatError, match="must be 2D"):
        validate_trajectory_array(arr, source="bad.csv")


def test_trajectory_too_few_columns() -> None:
    arr = np.array([[0.0], [0.1]])
    with pytest.raises(DataFormatError, match="too few columns"):
        validate_trajectory_array(arr, source="bad.csv", min_columns=2)


def test_trajectory_rejects_non_finite() -> None:
    arr = np.array([[0.0, 1.0], [0.1, np.nan]])
    with pytest.raises(DataFormatError, match="non-finite"):
        validate_trajectory_array(arr, source="bad.csv")


def test_trajectory_rejects_non_numeric() -> None:
    arr = np.array([["0.0", "a"], ["0.1", "b"]], dtype=object)
    with pytest.raises(DataFormatError, match="must be numeric"):
        validate_trajectory_array(arr, source="bad.csv")


def test_none_array_raises_value_error() -> None:
    with pytest.raises(ValueError, match="array must be provided"):
        validate_trajectory_array(None, source="x.csv")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# pandera availability probe
# ---------------------------------------------------------------------------


def test_pandera_available_returns_bool() -> None:
    assert isinstance(pandera_available(), bool)
