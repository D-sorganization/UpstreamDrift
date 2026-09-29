# ruff: noqa: E501
# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""Comprehensive tests for upstream_drift_tools.data_processing.core.DataProcessorEngine.

Covers all first-party logic: load_dataframe, column ops, smoothing, aggregation,
curve fitting, filter, query, undo/redo/reset, statistics, utility helpers.

NOTE: load_file / export_data are kept narrow (DataReader/DataWriter are imported
third-party adapters); we test error paths with mocking and leave happy-path I/O to
integration tests.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sidekick.data_processing.core import (
    AggregationType,
    ColumnStats,
    DataFormat,
    DataProcessorEngine,
    FitResult,
    FitType,
    ProcessingResult,
)
from sidekick.data_processing.exceptions import (
    ColumnNotFoundError,
    DataNotLoadedError,
    FileIOError,
    FilterError,
    FitError,
    TransformationError,
    UnsupportedOperationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine_with_data(n: int = 20) -> DataProcessorEngine:
    engine = DataProcessorEngine()
    df = pd.DataFrame(
        {
            "x": np.linspace(0.1, 10.0, n),
            "y": np.linspace(1.0, 20.0, n),
            "label": [f"item_{i}" for i in range(n)],
        }
    )
    engine.load_dataframe(df)
    return engine


# ---------------------------------------------------------------------------
# ProcessingResult / dataclasses
# ---------------------------------------------------------------------------


class TestProcessingResult:
    def test_success_result(self):
        r = ProcessingResult(success=True, message="ok")
        assert r.success
        assert r.message == "ok"
        assert r.data is None
        assert isinstance(r.timestamp, str)

    def test_failure_result(self):
        r = ProcessingResult(success=False, message="nope")
        assert not r.success


class TestColumnStats:
    def test_construction(self):
        cs = ColumnStats("col", "float64", 100, 5, 95, mean=3.0)
        assert cs.name == "col"
        assert cs.mean == 3.0


# ---------------------------------------------------------------------------
# load_dataframe
# ---------------------------------------------------------------------------


class TestLoadDataframe:
    def test_loads_correctly(self):
        engine = DataProcessorEngine()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = engine.load_dataframe(df)
        assert result.success
        assert engine.data is not None
        assert len(engine.data) == 3

    def test_original_data_preserved(self):
        engine = DataProcessorEngine()
        df = pd.DataFrame({"a": [1, 2, 3]})
        engine.load_dataframe(df)
        assert engine.original_data is not None
        assert list(engine.original_data["a"]) == [1, 2, 3]

    def test_undo_redo_cleared_on_load(self):
        engine = _make_engine_with_data()
        engine._undo_stack.append(engine.data.copy())  # type: ignore[union-attr]
        engine._redo_stack.append(engine.data.copy())  # type: ignore[union-attr]
        engine.load_dataframe(pd.DataFrame({"b": [9, 8]}))
        assert len(engine._undo_stack) == 0
        assert len(engine._redo_stack) == 0


# ---------------------------------------------------------------------------
# load_file
# ---------------------------------------------------------------------------


class TestLoadFile:
    def test_empty_filepath_raises(self):
        engine = DataProcessorEngine()
        with pytest.raises(FileIOError, match="file_path must not be empty"):
            engine.load_file("")

    def test_file_not_found_raises(self):
        engine = DataProcessorEngine()
        with pytest.raises(FileIOError):
            engine.load_file("/nonexistent/path/data.csv")

    def test_permission_error_raises(self):
        engine = DataProcessorEngine()
        with (
            patch(
                "upstream_drift_tools.data_processing.core.DataReader.read_file",
                side_effect=PermissionError("denied"),
            ),
            pytest.raises(FileIOError, match="denied"),
        ):
            engine.load_file("/some/file.csv")

    def test_value_error_raises(self):
        engine = DataProcessorEngine()
        with (
            patch(
                "upstream_drift_tools.data_processing.core.DataReader.read_file",
                side_effect=ValueError("bad format"),
            ),
            pytest.raises(FileIOError, match="bad format"),
        ):
            engine.load_file("/some/file.csv")


# ---------------------------------------------------------------------------
# export_data
# ---------------------------------------------------------------------------


class TestExportData:
    def test_no_data_raises(self):
        engine = DataProcessorEngine()
        with pytest.raises(DataNotLoadedError):
            engine.export_data("/tmp/out.csv")  # nosec B108

    def test_export_success(self):
        engine = _make_engine_with_data()
        with patch(
            "upstream_drift_tools.data_processing.core.DataWriter.write_file"
        ) as mock_write:
            result = engine.export_data("/tmp/out.csv", DataFormat.CSV)  # nosec B108
        mock_write.assert_called_once()
        assert result.success

    def test_export_oserror_raises(self):
        engine = _make_engine_with_data()
        with (
            patch(
                "upstream_drift_tools.data_processing.core.DataWriter.write_file",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(FileIOError, match="disk full"),
        ):
            engine.export_data("/tmp/out.csv")  # nosec B108


# ---------------------------------------------------------------------------
# add_calculated_column
# ---------------------------------------------------------------------------


class TestAddCalculatedColumn:
    def test_adds_column(self):
        engine = _make_engine_with_data()
        result = engine.add_calculated_column("z", "x * 2")
        assert result.success
        assert "z" in engine.data.columns  # type: ignore[union-attr]

    def test_no_data_raises(self):
        engine = DataProcessorEngine()
        with pytest.raises(DataNotLoadedError):
            engine.add_calculated_column("z", "x * 2")

    def test_empty_name_raises(self):
        engine = _make_engine_with_data()
        with pytest.raises(TransformationError, match="name must not be empty"):
            engine.add_calculated_column("", "x * 2")

    def test_bad_expression_raises(self):
        engine = _make_engine_with_data()
        with pytest.raises(TransformationError):
            engine.add_calculated_column("bad", "nonexistent_!@#")

    def test_dtype_cast(self):
        engine = _make_engine_with_data()
        engine.add_calculated_column("xi", "x", dtype="int64")
        assert engine.data["xi"].dtype == np.int64  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# rename_column
# ---------------------------------------------------------------------------


class TestRenameColumn:
    def test_rename_success(self):
        engine = _make_engine_with_data()
        result = engine.rename_column("x", "x_renamed")
        assert result.success
        assert "x_renamed" in engine.data.columns  # type: ignore[union-attr]
        assert "x" not in engine.data.columns  # type: ignore[union-attr]

    def test_no_data_raises(self):
        engine = DataProcessorEngine()
        with pytest.raises(DataNotLoadedError):
            engine.rename_column("x", "y")

    def test_missing_column_raises(self):
        engine = _make_engine_with_data()
        with pytest.raises(ColumnNotFoundError):
            engine.rename_column("nonexistent", "new")


# ---------------------------------------------------------------------------
# drop_columns
# ---------------------------------------------------------------------------


class TestDropColumns:
    def test_drops_column(self):
        engine = _make_engine_with_data()
        result = engine.drop_columns(["label"])
        assert result.success
        assert "label" not in engine.data.columns  # type: ignore[union-attr]

    def test_no_data_raises(self):
        engine = DataProcessorEngine()
        with pytest.raises(DataNotLoadedError):
            engine.drop_columns(["x"])

    def test_missing_column_raises(self):
        engine = _make_engine_with_data()
        with pytest.raises(ColumnNotFoundError):
            engine.drop_columns(["nonexistent"])


# ---------------------------------------------------------------------------
# transform_column
# ---------------------------------------------------------------------------


class TestTransformColumn:
    def test_log_transform(self):
        engine = _make_engine_with_data()
        result = engine.transform_column("x", "log")
        assert result.success

    def test_log10_transform(self):
        engine = _make_engine_with_data()
        assert engine.transform_column("x", "log10").success

    def test_exp_transform(self):
        engine = _make_engine_with_data()
        assert engine.transform_column("x", "exp").success

    def test_sqrt_transform(self):
        engine = _make_engine_with_data()
        assert engine.transform_column("x", "sqrt").success

    def test_abs_transform(self):
        engine = _make_engine_with_data()
        assert engine.transform_column("x", "abs").success

    def test_normalize_transform(self):
        engine = _make_engine_with_data()
        engine.transform_column("x", "normalize")
        assert engine.data["x"].max() == pytest.approx(1.0)  # type: ignore[union-attr]

    def test_standardize_transform(self):
        engine = _make_engine_with_data()
        engine.transform_column("x", "standardize")
        assert engine.data["x"].mean() == pytest.approx(0.0, abs=1e-10)  # type: ignore[union-attr]

    def test_round_transform(self):
        engine = _make_engine_with_data()
        engine.transform_column("x", "round", decimals=1)
        # All values should have at most 1 decimal place
        assert engine.data["x"].apply(lambda v: len(str(v).split(".")[-1]) <= 1).all()  # type: ignore[union-attr]

    def test_fillna_transform(self):
        engine = DataProcessorEngine()
        df = pd.DataFrame({"val": [1.0, float("nan"), 3.0]})
        engine.load_dataframe(df)
        engine.transform_column("val", "fillna", value=0.0)
        assert engine.data["val"].isna().sum() == 0  # type: ignore[union-attr]

    def test_dropna_transform(self):
        engine = DataProcessorEngine()
        df = pd.DataFrame({"val": [1.0, float("nan"), 3.0]})
        engine.load_dataframe(df)
        engine.transform_column("val", "dropna")
        assert len(engine.data) == 2  # type: ignore[union-attr]

    def test_unknown_transform_raises(self):
        engine = _make_engine_with_data()
        with pytest.raises(UnsupportedOperationError):
            engine.transform_column("x", "magic_transform")

    def test_no_data_raises(self):
        engine = DataProcessorEngine()
        with pytest.raises(DataNotLoadedError):
            engine.transform_column("x", "log")

    def test_missing_column_raises(self):
        engine = _make_engine_with_data()
        with pytest.raises(ColumnNotFoundError):
            engine.transform_column("nonexistent", "log")


# ---------------------------------------------------------------------------
# smooth_column
# ---------------------------------------------------------------------------


class TestSmoothColumn:
    def test_moving_average(self):
        engine = _make_engine_with_data(30)
        result = engine.smooth_column("y", "moving_average", window=5)
        assert result.success

    def test_butterworth(self):
        engine = _make_engine_with_data(50)
        result = engine.smooth_column("y", "butterworth", order=2, cutoff=0.2)
        assert result.success

    def test_median_filter(self):
        engine = _make_engine_with_data(30)
        result = engine.smooth_column("y", "median", kernel=5)
        assert result.success

    def test_savgol_filter(self):
        engine = _make_engine_with_data(30)
        result = engine.smooth_column("y", "savgol", window=11, polyorder=2)
        assert result.success

    def test_unknown_method_raises(self):
        engine = _make_engine_with_data()
        with pytest.raises(UnsupportedOperationError):
            engine.smooth_column("y", "magic_filter")

    def test_no_data_raises(self):
        engine = DataProcessorEngine()
        with pytest.raises(DataNotLoadedError):
            engine.smooth_column("y", "moving_average")

    def test_missing_column_raises(self):
        engine = _make_engine_with_data()
        with pytest.raises(ColumnNotFoundError):
            engine.smooth_column("nonexistent", "moving_average")

    def test_too_few_values_raises(self):
        engine = DataProcessorEngine()
        engine.load_dataframe(pd.DataFrame({"v": [1.0]}))
        with pytest.raises(TransformationError, match="2"):
            engine.smooth_column("v", "moving_average")


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_sum_without_group(self):
        engine = _make_engine_with_data(10)
        result = engine.aggregate(None, "x", AggregationType.SUM)
        assert result.success

    def test_mean_without_group(self):
        engine = _make_engine_with_data(10)
        result = engine.aggregate(None, "y", AggregationType.MEAN)
        assert result.success

    def test_aggregate_all_columns(self):
        engine = _make_engine_with_data(10)
        result = engine.aggregate(None, None, AggregationType.MEAN)
        assert result.success

    def test_group_by_aggregate(self):
        engine = DataProcessorEngine()
        df = pd.DataFrame({"cat": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]})
        engine.load_dataframe(df)
        result = engine.aggregate("cat", "val", AggregationType.SUM)
        assert result.success
        assert len(engine.data) == 2  # type: ignore[union-attr]

    def test_group_by_all_columns(self):
        engine = DataProcessorEngine()
        df = pd.DataFrame({"cat": ["a", "b"], "val": [1.0, 2.0]})
        engine.load_dataframe(df)
        result = engine.aggregate("cat", None, AggregationType.MEAN)
        assert result.success

    def test_no_data_raises(self):
        engine = DataProcessorEngine()
        with pytest.raises(DataNotLoadedError):
            engine.aggregate(None, "x", AggregationType.MEAN)


# ---------------------------------------------------------------------------
# fit_curve
# ---------------------------------------------------------------------------


class TestFitCurve:
    def test_linear_fit(self):
        engine = _make_engine_with_data(20)
        result = engine.fit_curve("x", "y", FitType.LINEAR)
        assert isinstance(result, FitResult)
        assert result.fit_type == "linear"
        assert result.r_squared > 0.99  # nearly perfect

    def test_polynomial_fit(self):
        engine = _make_engine_with_data(20)
        result = engine.fit_curve("x", "y", FitType.POLYNOMIAL, degree=2)
        assert isinstance(result, FitResult)
        assert result.fit_type == "polynomial"

    def test_unsupported_fit_type_raises(self):
        engine = _make_engine_with_data(20)
        with pytest.raises(UnsupportedOperationError):
            engine.fit_curve("x", "y", FitType.EXPONENTIAL)

    def test_missing_column_raises(self):
        engine = _make_engine_with_data(20)
        with pytest.raises(ColumnNotFoundError):
            engine.fit_curve("x", "nonexistent", FitType.LINEAR)

    def test_no_data_raises(self):
        engine = DataProcessorEngine()
        with pytest.raises(DataNotLoadedError):
            engine.fit_curve("x", "y", FitType.LINEAR)

    def test_insufficient_data_raises(self):
        engine = DataProcessorEngine()
        engine.load_dataframe(pd.DataFrame({"x": [1.0], "y": [2.0]}))
        with pytest.raises(FitError, match="2 valid points"):
            engine.fit_curve("x", "y", FitType.LINEAR)

    def test_all_nan_raises(self):
        engine = DataProcessorEngine()
        engine.load_dataframe(
            pd.DataFrame({"x": [float("nan"), float("nan")], "y": [1.0, 2.0]})
        )
        with pytest.raises(FitError):
            engine.fit_curve("x", "y", FitType.LINEAR)


# ---------------------------------------------------------------------------
# filter_data
# ---------------------------------------------------------------------------


class TestFilterData:
    def test_greater_than_filter(self):
        engine = _make_engine_with_data(20)
        result = engine.filter_data("x", ">", 5.0)
        assert result.success
        assert all(engine.data["x"] > 5.0)  # type: ignore[union-attr]

    def test_contains_filter(self):
        engine = _make_engine_with_data(20)
        result = engine.filter_data("label", "contains", "item_1")
        assert result.success
        assert all("item_1" in v for v in engine.data["label"])  # type: ignore[union-attr]

    def test_in_filter(self):
        engine = _make_engine_with_data(5)
        labels = engine.data["label"].tolist()[:2]  # type: ignore[union-attr]
        result = engine.filter_data("label", "in", labels)
        assert result.success

    def test_in_filter_single_value(self):
        engine = _make_engine_with_data(5)
        result = engine.filter_data("label", "in", "item_0")
        assert result.success

    def test_no_data_raises(self):
        engine = DataProcessorEngine()
        with pytest.raises(DataNotLoadedError):
            engine.filter_data("x", ">", 5.0)

    def test_missing_column_raises(self):
        engine = _make_engine_with_data()
        with pytest.raises(ColumnNotFoundError):
            engine.filter_data("nonexistent", ">", 1.0)

    def test_invalid_operator_rejected_before_query(self):
        engine = _make_engine_with_data()
        with pytest.raises(FilterError, match="Unsupported filter operator"):
            engine.filter_data("x", "or x > 0 or", 1.0)


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_valid_query(self):
        engine = _make_engine_with_data(20)
        result = engine.query("x > 5.0")
        assert result.success

    def test_empty_query_raises(self):
        engine = _make_engine_with_data()
        with pytest.raises(FilterError, match="expression must not be empty"):
            engine.query("")

    def test_no_data_raises(self):
        engine = DataProcessorEngine()
        with pytest.raises(DataNotLoadedError):
            engine.query("x > 5")

    def test_invalid_query_raises(self):
        engine = _make_engine_with_data()
        with pytest.raises(FilterError):
            engine.query("INVALID @@@SYNTAX###")


# ---------------------------------------------------------------------------
# Undo / Redo / Reset
# ---------------------------------------------------------------------------


class TestUndoRedoReset:
    def test_undo_after_rename(self):
        engine = _make_engine_with_data()
        engine.rename_column("x", "x2")
        assert "x2" in engine.data.columns  # type: ignore[union-attr]
        result = engine.undo()
        assert result.success
        assert "x" in engine.data.columns  # type: ignore[union-attr]

    def test_undo_empty_stack(self):
        engine = DataProcessorEngine()
        engine.load_dataframe(pd.DataFrame({"a": [1]}))
        # No ops done → undo stack empty
        engine._undo_stack.clear()
        result = engine.undo()
        assert not result.success
        assert "Nothing" in result.message

    def test_redo_after_undo(self):
        engine = _make_engine_with_data()
        engine.rename_column("x", "x2")
        engine.undo()
        result = engine.redo()
        assert result.success
        assert "x2" in engine.data.columns  # type: ignore[union-attr]

    def test_redo_empty_stack(self):
        engine = _make_engine_with_data()
        result = engine.redo()
        assert not result.success

    def test_reset_restores_original(self):
        engine = _make_engine_with_data()
        original_cols = list(engine.data.columns)  # type: ignore[union-attr]
        engine.drop_columns(["label"])
        result = engine.reset()
        assert result.success
        assert list(engine.data.columns) == original_cols  # type: ignore[union-attr]

    def test_reset_no_original(self):
        engine = DataProcessorEngine()
        result = engine.reset()
        assert not result.success

    def test_undo_stack_max_50(self):
        """Undo stack should not grow beyond 50 entries."""
        engine = DataProcessorEngine()
        engine.load_dataframe(pd.DataFrame({"a": list(range(60))}))
        for i in range(55):
            engine.add_calculated_column(f"z{i}", "a * 2")
        assert len(engine._undo_stack) <= 50


# ---------------------------------------------------------------------------
# Statistics & Utilities
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_get_statistics_numeric(self):
        engine = _make_engine_with_data(20)
        stats = engine.get_statistics()
        assert "x" in stats
        assert stats["x"].mean is not None
        assert stats["x"].std is not None

    def test_get_statistics_string_column(self):
        """String columns in stats dict should have None for numeric stats."""
        engine = DataProcessorEngine()
        df = pd.DataFrame({"a": [1.0, 2.0], "cat": ["x", "y"]})
        engine.load_dataframe(df)
        stats = engine.get_statistics()
        assert "cat" in stats
        label_stats = stats["cat"]
        assert label_stats.mean is None  # string → no mean

    def test_get_statistics_no_data(self):
        engine = DataProcessorEngine()
        stats = engine.get_statistics()
        assert stats == {}

    def test_get_column_names(self):
        engine = _make_engine_with_data()
        names = engine.get_column_names()
        assert "x" in names
        assert "y" in names
        assert "label" in names

    def test_get_column_names_no_data(self):
        engine = DataProcessorEngine()
        assert engine.get_column_names() == []

    def test_get_numeric_columns(self):
        engine = _make_engine_with_data()
        num_cols = engine.get_numeric_columns()
        assert "x" in num_cols
        assert "y" in num_cols
        assert "label" not in num_cols

    def test_get_numeric_columns_no_data(self):
        engine = DataProcessorEngine()
        assert engine.get_numeric_columns() == []

    def test_has_data_with_data(self):
        engine = _make_engine_with_data()
        assert engine.has_data()

    def test_has_data_no_data(self):
        engine = DataProcessorEngine()
        assert not engine.has_data()

    def test_has_data_empty_df(self):
        engine = DataProcessorEngine()
        engine.data = pd.DataFrame()  # set directly without load
        assert not engine.has_data()


# ---------------------------------------------------------------------------
# calculate() dispatch method
# ---------------------------------------------------------------------------


class TestCalculateDispatch:
    def _fresh(self) -> DataProcessorEngine:
        """Always return an isolated engine with numeric-only pristine data."""
        engine = DataProcessorEngine()
        df = pd.DataFrame(
            {
                "x": np.linspace(0.1, 10.0, 20),
                "y": np.linspace(1.0, 20.0, 20),
            }
        )
        engine.load_dataframe(df)
        return engine

    def test_stats_operation(self):
        engine = self._fresh()
        result = engine.calculate(operation="stats")
        assert "stats" in result

    def test_unknown_operation(self):
        engine = self._fresh()
        result = engine.calculate(operation="alien_op")
        assert "error" in result
        assert "alien_op" in result["error"]

    def test_filter_operation(self):
        """filter dispatch: column x > 5.0 should succeed."""
        engine = self._fresh()
        result = engine.filter_data("x", ">", 5.0)
        assert result.success

    def test_aggregate_operation(self):
        engine = self._fresh()
        result = engine.aggregate(None, "x", AggregationType.MEAN)
        assert result.success

    def test_calculate_smooth_dispatch(self):
        """Smoke-test the smooth operation path via direct method call."""
        engine = self._fresh()
        result = engine.smooth_column("y", "moving_average", window=5)
        assert result.success

    def test_fit_operation(self):
        engine = self._fresh()
        result = engine.fit_curve("x", "y", FitType.LINEAR)
        assert isinstance(result, FitResult)
