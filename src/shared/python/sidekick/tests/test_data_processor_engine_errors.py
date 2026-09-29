"""Additional tests for DataProcessorEngine error paths not covered in the main suite.

Targets remaining uncovered lines in core.py:
- Line 172: _wrap_result (covered via calculate)
- Lines 191-196: load_file exception path
- Lines 337-339: transform ValueError/TypeError → undo + TransformationError
- Lines 404-406: smooth ValueError/TypeError → TransformationError
- Lines 442-444: aggregate KeyError/TypeError → TransformationError
- Lines 469-470: fit_curve non-numeric data → FitError
- Lines 526-528: filter_data exception → FilterError
- Line 572: query exception path
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from sidekick.data_processing.core import (
    AggregationType,
    DataProcessorEngine,
    FitType,
    ProcessingResult,
)
from sidekick.data_processing.exceptions import (
    ColumnNotFoundError,
    DataProcessingError,
    FileIOError,
    FilterError,
    FitError,
    TransformationError,
)


def _engine_with_data(**extra_cols: object) -> DataProcessorEngine:
    """Helper: create a fresh DataProcessorEngine with a simple DataFrame loaded."""
    engine = DataProcessorEngine()
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]})
    for col, vals in extra_cols.items():
        df[col] = vals
    engine.load_dataframe(df)
    return engine


# ---------------------------------------------------------------------------
# _wrap_result (line 172)
# ---------------------------------------------------------------------------


class TestWrapResult:
    def test_wrap_result_returns_correct_dict(self) -> None:
        """_wrap_result converts ProcessingResult to dict with the correct keys."""
        engine = DataProcessorEngine()
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = ProcessingResult(success=True, message="OK", data=df, stats={})
        wrapped = engine._wrap_result(result)
        assert "success" in wrapped
        assert wrapped["success"] is True
        assert wrapped["message"] == "OK"
        assert "stats" in wrapped
        assert "timestamp" in wrapped


# ---------------------------------------------------------------------------
# load_file error path (lines 191-196)
# ---------------------------------------------------------------------------


class TestLoadFileErrorPath:
    def test_load_file_with_invalid_format_raises(self) -> None:
        """DataReader.read_file raising ValueError → re-raises as FileIOError."""
        engine = DataProcessorEngine()
        with (
            patch(
                "upstream_drift_tools.data_processing.core.DataReader.read_file",
                side_effect=ValueError("unsupported format"),
            ),
            pytest.raises((FileIOError, ValueError)),
        ):
            engine.load_file("/tmp/bad.xyz")  # nosec B108

    def test_load_file_empty_path_raises(self) -> None:
        engine = DataProcessorEngine()
        with pytest.raises(FileIOError, match="file_path must not be empty"):
            engine.load_file("")

    def test_load_file_success_path(self) -> None:
        """Lines 191-196: successful load_file sets data, original_data, path."""
        engine = DataProcessorEngine()
        mock_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        with patch(
            "upstream_drift_tools.data_processing.core.DataReader.read_file",
            return_value=mock_df,
        ):
            result = engine.load_file("/some/file.csv")

        # load_file returns a ProcessingResult object
        assert result.success is True
        assert engine.data is not None
        assert len(engine.data) == 2
        assert engine.file_path is not None


# ---------------------------------------------------------------------------
# _get_basic_stats with data=None (line 572)
# ---------------------------------------------------------------------------


class TestGetBasicStatsNull:
    def test_get_basic_stats_returns_empty_when_no_data(self) -> None:
        """Line 572: _get_basic_stats returns {} when self.data is None."""
        engine = DataProcessorEngine()
        # data is None at start
        assert engine.data is None
        result = engine._get_basic_stats()
        assert result == {}


# ---------------------------------------------------------------------------
# transform_column error path (lines 337-339)
# ---------------------------------------------------------------------------


class TestTransformColumnErrorPath:
    def test_transform_invalid_column_raises_column_not_found(self) -> None:
        engine = _engine_with_data()
        with pytest.raises(ColumnNotFoundError):
            engine.transform_column("nonexistent_col", "log")

    def test_transform_value_error_raises_transformation_error(self) -> None:
        """TypeError/ValueError during transform → undo + TransformationError."""
        engine = _engine_with_data()
        # Patching np.log to raise TypeError to trigger lines 337-339
        with patch("numpy.log", side_effect=TypeError("type error")):
            with pytest.raises(TransformationError, match="type error"):
                engine.transform_column("x", "log")


# ---------------------------------------------------------------------------
# smooth_column error path (lines 404-406)
# ---------------------------------------------------------------------------


class TestSmoothColumnErrorPath:
    def test_smooth_value_error_raises_transformation_error(self) -> None:
        """ValueError during moving_average → undo + TransformationError."""
        engine = _engine_with_data()
        # Patch rolling().mean() to raise ValueError
        with (
            patch(
                "pandas.core.window.rolling.Rolling.mean",
                side_effect=ValueError("bad window"),
            ),
            pytest.raises(TransformationError),
        ):
            engine.smooth_column("x", "moving_average", window=5)


# ---------------------------------------------------------------------------
# aggregate error path (lines 442-444)
# ---------------------------------------------------------------------------


class TestAggregateErrorPath:
    def test_aggregate_missing_column_raises_transformation_error(self) -> None:
        """aggregate() with unknown column raises TransformationError."""
        engine = _engine_with_data()
        with pytest.raises(TransformationError, match="Aggregation failed"):
            engine.aggregate(
                group_by=None, column="nonexistent_xyz", agg_type=AggregationType.MEAN
            )


# ---------------------------------------------------------------------------
# fit_curve non-numeric data (lines 469-470)
# ---------------------------------------------------------------------------


class TestFitCurveErrorPath:
    def test_fit_curve_non_numeric_raises_fit_error(self) -> None:
        """Non-numeric column data raises FitError."""
        engine = DataProcessorEngine()
        df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})
        engine.load_dataframe(df)

        with pytest.raises(FitError, match="Non-numeric"):
            engine.fit_curve("x", "y", FitType.LINEAR)


# ---------------------------------------------------------------------------
# filter_data error path (lines 526-528)
# ---------------------------------------------------------------------------


class TestFilterDataErrorPath:
    def test_filter_query_expression_error_raises_filter_error(self) -> None:
        """A bad pandas query expression raises FilterError."""
        engine = _engine_with_data()
        with pytest.raises(FilterError):
            engine.filter_data("x", ">", "not_a_number")

    def test_filter_bad_query_operator_raises_filter_error(self) -> None:
        """Using a pandas query op with a non-numeric value raises FilterError."""
        engine = _engine_with_data()
        # query() raises on invalid expression syntax → FilterError
        with pytest.raises(FilterError):
            engine.filter_data("x", "??", 1.0)  # invalid operator in query


# ---------------------------------------------------------------------------
# query exception path (line 572)
# ---------------------------------------------------------------------------


class TestQueryErrorPath:
    def test_query_bad_expression_raises_filter_error(self) -> None:
        engine = _engine_with_data()
        with pytest.raises(FilterError):
            engine.query("this is not valid pandas query syntax @@@")

    def test_query_empty_expression_raises_filter_error(self) -> None:
        engine = _engine_with_data()
        with pytest.raises(FilterError, match="expression"):
            engine.query("")


# ---------------------------------------------------------------------------
# ColumnNotFoundError — direct tests for __init__ branches (lines 22-28)
# ---------------------------------------------------------------------------


class TestColumnNotFoundError:
    def test_without_available_list(self) -> None:
        """Lines 22-25: no available columns → message without suffix."""
        err = ColumnNotFoundError("missing_col")
        assert err.column == "missing_col"
        assert err.available == []
        assert "missing_col" in str(err)
        assert "Available" not in str(err)

    def test_with_available_list(self) -> None:
        """Lines 26-27: available columns → message with 'Available' suffix."""
        err = ColumnNotFoundError("missing_col", ["col_a", "col_b"])
        assert err.column == "missing_col"
        assert err.available == ["col_a", "col_b"]
        assert "Available" in str(err)
        assert "col_a" in str(err)

    def test_is_data_processing_error(self) -> None:
        """ColumnNotFoundError is a DataProcessingError subclass."""
        err = ColumnNotFoundError("x")
        assert isinstance(err, DataProcessingError)
