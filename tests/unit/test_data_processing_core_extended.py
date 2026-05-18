"""Extended unit tests for DataProcessorEngine and helpers in data_processing/core.py.

Tests cover:
- Enums: AggregationType, DataFormat, FitType
- Dataclasses: ColumnStats, ProcessingResult, FitResult
- Exception hierarchy
- DataProcessorEngine: load_dataframe, has_data, get_column_names,
  get_numeric_columns, get_statistics, filter_data, aggregate,
  transform_column, add_calculated_column, fit_curve, smooth_column,
  query, rename_column, drop_columns, reset, undo/redo
- Error cases: DataNotLoadedError, ColumnNotFoundError, etc.

All tests are headless-safe with no heavy dependencies beyond pandas/numpy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from sidekick.data_processing.core import (
        DataProcessorEngine,
    )

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 5) -> pd.DataFrame:
    """Return a simple numeric DataFrame with columns a, b."""
    return pd.DataFrame(
        {
            "a": [float(i) for i in range(1, n + 1)],
            "b": [float(i * 2) for i in range(1, n + 1)],
        }
    )


def _make_df_with_group(n: int = 4) -> pd.DataFrame:
    """Return a DataFrame with numeric columns and a group column."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "grp": ["x", "x", "y", "y"],
        }
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestAggregationType:
    """Tests for AggregationType enum."""

    def test_has_expected_members(self) -> None:
        """AggregationType has SUM, MEAN, MEDIAN, STD, MIN, MAX, COUNT."""
        from sidekick.data_processing.core import (
            AggregationType,
        )

        assert AggregationType.SUM.value == "sum"
        assert AggregationType.MEAN.value == "mean"
        assert AggregationType.MEDIAN.value == "median"
        assert AggregationType.COUNT.value == "count"

    def test_at_least_six_members(self) -> None:
        """AggregationType has at least 6 variants."""
        from sidekick.data_processing.core import (
            AggregationType,
        )

        assert len(list(AggregationType)) >= 6


class TestDataFormat:
    """Tests for DataFormat enum."""

    def test_csv_and_json(self) -> None:
        """DataFormat includes CSV and JSON."""
        from sidekick.data_processing.core import (
            DataFormat,
        )

        assert DataFormat.CSV.value == "csv"
        assert DataFormat.JSON.value == "json"


class TestFitType:
    """Tests for FitType enum."""

    def test_linear_and_polynomial(self) -> None:
        """FitType includes LINEAR and POLYNOMIAL."""
        from sidekick.data_processing.core import FitType

        assert FitType.LINEAR.value == "linear"
        assert FitType.POLYNOMIAL.value == "polynomial"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptions:
    """Tests for data_processing exception hierarchy."""

    def test_data_not_loaded_is_base(self) -> None:
        """DataNotLoadedError can be raised and caught as a DataProcessingError."""
        from sidekick.data_processing.core import (
            DataNotLoadedError,
        )
        from sidekick.data_processing.exceptions import (
            DataProcessingError,
        )

        exc = DataNotLoadedError("no data")
        assert isinstance(exc, DataProcessingError)

    def test_column_not_found_carries_column_name(self) -> None:
        """ColumnNotFoundError stores the missing column name."""
        from sidekick.data_processing.core import (
            ColumnNotFoundError,
        )

        exc = ColumnNotFoundError("missing_col", ["a", "b"])
        assert "missing_col" in str(exc)

    def test_filter_error_is_base(self) -> None:
        """FilterError is a DataProcessingError."""
        from sidekick.data_processing.core import (
            FilterError,
        )
        from sidekick.data_processing.exceptions import (
            DataProcessingError,
        )

        exc = FilterError("bad filter")
        assert isinstance(exc, DataProcessingError)

    def test_transformation_error_is_base(self) -> None:
        """TransformationError is a DataProcessingError."""
        from sidekick.data_processing.core import (
            TransformationError,
        )
        from sidekick.data_processing.exceptions import (
            DataProcessingError,
        )

        exc = TransformationError("bad transform")
        assert isinstance(exc, DataProcessingError)

    def test_fit_error_is_base(self) -> None:
        """FitError is a DataProcessingError."""
        from sidekick.data_processing.core import FitError
        from sidekick.data_processing.exceptions import (
            DataProcessingError,
        )

        exc = FitError("fit failed")
        assert isinstance(exc, DataProcessingError)

    def test_unsupported_operation_is_base(self) -> None:
        """UnsupportedOperationError is a DataProcessingError."""
        from sidekick.data_processing.core import (
            UnsupportedOperationError,
        )
        from sidekick.data_processing.exceptions import (
            DataProcessingError,
        )

        exc = UnsupportedOperationError("nope")
        assert isinstance(exc, DataProcessingError)


# ---------------------------------------------------------------------------
# ColumnStats dataclass
# ---------------------------------------------------------------------------


class TestColumnStats:
    """Tests for the ColumnStats dataclass."""

    def test_data_processing_core_extended_instantiation(self) -> None:
        """ColumnStats can be instantiated with all fields."""
        from sidekick.data_processing.core import (
            ColumnStats,
        )

        cs = ColumnStats(
            name="x",
            dtype="float64",
            count=10,
            null_count=0,
            unique_count=10,
            mean=5.0,
            std=2.0,
            min_val=1.0,
            max_val=9.0,
            median=5.0,
            q25=3.0,
            q75=7.0,
        )
        assert cs.name == "x"
        assert cs.count == 10


# ---------------------------------------------------------------------------
# ProcessingResult dataclass
# ---------------------------------------------------------------------------


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_success_field(self) -> None:
        """ProcessingResult stores success and message."""
        from sidekick.data_processing.core import (
            ProcessingResult,
        )

        r = ProcessingResult(success=True, message="ok")
        assert r.solver_status == "success"
        assert r.message == "ok"

    def test_default_data_and_stats(self) -> None:
        """ProcessingResult data defaults to None; stats defaults to empty dict."""
        from sidekick.data_processing.core import (
            ProcessingResult,
        )

        r = ProcessingResult(success=False, message="fail")
        assert r.data is None
        # stats defaults to empty dict
        assert r.stats == {}


# ---------------------------------------------------------------------------
# FitResult dataclass
# ---------------------------------------------------------------------------


class TestFitResult:
    """Tests for FitResult dataclass."""

    def test_data_processing_core_extended_instantiation(self) -> None:
        """FitResult can be constructed with required fields."""
        from sidekick.data_processing.core import (
            FitResult,
            FitType,
        )

        fr = FitResult(
            fit_type=FitType.LINEAR.value,
            coefficients=[2.0, 0.0],
            r_squared=1.0,
            equation="y = 2x",
            fitted_values=np.array([2.0, 4.0]),
            residuals=np.array([0.0, 0.0]),
        )
        assert fr.fit_type == FitType.LINEAR.value
        assert fr.r_squared == 1.0


# ---------------------------------------------------------------------------
# DataProcessorEngine — initialization
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> DataProcessorEngine:
    """Fresh DataProcessorEngine for each test."""
    from sidekick.data_processing.core import (
        DataProcessorEngine,
    )

    return DataProcessorEngine()


@pytest.fixture
def loaded_engine() -> DataProcessorEngine:
    """DataProcessorEngine with a 5-row DataFrame already loaded."""
    from sidekick.data_processing.core import (
        DataProcessorEngine,
    )

    e = DataProcessorEngine()
    e.load_dataframe(_make_df(5))
    return e


class TestDataProcessorEngineInit:
    """Tests for DataProcessorEngine initialization."""

    def test_instantiates_successfully(self, engine) -> None:
        """Engine can be created without errors."""
        assert engine is not None

    def test_has_data_false_initially(self, engine) -> None:
        """has_data() returns False before loading data."""
        assert engine.has_data() is False

    def test_get_column_names_empty_before_load(self, engine) -> None:
        """get_column_names() returns empty list before loading."""
        assert engine.get_column_names() == []

    def test_get_numeric_columns_empty_before_load(self, engine) -> None:
        """get_numeric_columns() returns empty list before loading."""
        assert engine.get_numeric_columns() == []


# ---------------------------------------------------------------------------
# DataProcessorEngine — load_dataframe
# ---------------------------------------------------------------------------


class TestLoadDataframe:
    """Tests for DataProcessorEngine.load_dataframe."""

    def test_has_data_after_load(self, engine) -> None:
        """has_data() returns True after loading a DataFrame."""
        engine.load_dataframe(_make_df())
        assert engine.has_data() is True

    def test_columns_available_after_load(self, engine) -> None:
        """get_column_names() returns the loaded DataFrame columns."""
        engine.load_dataframe(_make_df())
        assert "a" in engine.get_column_names()
        assert "b" in engine.get_column_names()

    def test_numeric_columns_detected(self, engine) -> None:
        """get_numeric_columns() returns numeric column names."""
        engine.load_dataframe(_make_df())
        numeric = engine.get_numeric_columns()
        assert "a" in numeric
        assert "b" in numeric


# ---------------------------------------------------------------------------
# DataProcessorEngine — get_statistics
# ---------------------------------------------------------------------------


class TestGetStatistics:
    """Tests for DataProcessorEngine.get_statistics."""

    def test_returns_dict_keyed_by_column(self, loaded_engine) -> None:
        """get_statistics() returns a dict with column names as keys."""
        stats = loaded_engine.get_statistics()
        assert isinstance(stats, dict)
        assert "a" in stats
        assert "b" in stats

    def test_stats_values_are_column_stats(self, loaded_engine) -> None:
        """Each stats value is a ColumnStats instance."""
        from sidekick.data_processing.core import (
            ColumnStats,
        )

        stats = loaded_engine.get_statistics()
        for val in stats.values():
            assert isinstance(val, ColumnStats)

    def test_count_matches_dataframe_length(self, loaded_engine) -> None:
        """ColumnStats.count matches the number of rows in the DataFrame."""
        stats = loaded_engine.get_statistics()
        assert stats["a"].count == 5


# ---------------------------------------------------------------------------
# DataProcessorEngine — filter_data
# ---------------------------------------------------------------------------


class TestFilterData:
    """Tests for DataProcessorEngine.filter_data."""

    def test_filter_gt_returns_filtered_rows(self, loaded_engine) -> None:
        """filter_data('a', '>', 2.0) returns rows where a > 2."""
        result = loaded_engine.filter_data("a", ">", 2.0)
        assert result.solver_status == "success"
        assert len(result.data) == 3  # rows with a=3,4,5

    def test_filter_eq_returns_matching_rows(self, loaded_engine) -> None:
        """filter_data('a', '==', 1.0) returns row where a == 1."""
        result = loaded_engine.filter_data("a", "==", 1.0)
        assert result.solver_status == "success"
        assert len(result.data) == 1

    def test_filter_lt_returns_filtered_rows(self, loaded_engine) -> None:
        """filter_data('a', '<', 3.0) returns rows where a < 3."""
        result = loaded_engine.filter_data("a", "<", 3.0)
        assert result.solver_status == "success"
        assert len(result.data) == 2

    def test_filter_no_data_raises(self, engine) -> None:
        """filter_data raises DataNotLoadedError when no data is loaded."""
        from sidekick.data_processing.core import (
            DataNotLoadedError,
        )

        with pytest.raises(DataNotLoadedError):
            engine.filter_data("a", ">", 1.0)

    def test_filter_missing_column_raises(self, loaded_engine) -> None:
        """filter_data raises ColumnNotFoundError for unknown column."""
        from sidekick.data_processing.core import (
            ColumnNotFoundError,
        )

        with pytest.raises(ColumnNotFoundError):
            loaded_engine.filter_data("z", ">", 1.0)


# ---------------------------------------------------------------------------
# DataProcessorEngine — aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    """Tests for DataProcessorEngine.aggregate."""

    def test_aggregate_mean_all_rows(self, loaded_engine) -> None:
        """aggregate(None, 'a', MEAN) computes global mean."""
        from sidekick.data_processing.core import (
            AggregationType,
        )

        result = loaded_engine.aggregate(None, "a", AggregationType.MEAN)
        assert result.solver_status == "success"

    def test_aggregate_sum_all_rows(self, loaded_engine) -> None:
        """aggregate(None, 'a', SUM) returns sum of column a."""
        from sidekick.data_processing.core import (
            AggregationType,
        )

        result = loaded_engine.aggregate(None, "a", AggregationType.SUM)
        assert result.solver_status == "success"

    def test_aggregate_by_group(self, engine) -> None:
        """aggregate('grp', 'a', SUM) groups by grp and sums a."""
        from sidekick.data_processing.core import (
            AggregationType,
        )

        engine.load_dataframe(_make_df_with_group())
        result = engine.aggregate("grp", "a", AggregationType.SUM)
        assert result.solver_status == "success"
        assert result.data is not None


# ---------------------------------------------------------------------------
# DataProcessorEngine — query
# ---------------------------------------------------------------------------


class TestQuery:
    """Tests for DataProcessorEngine.query."""

    def test_query_expression_filters(self, loaded_engine) -> None:
        """query('a > 2') returns rows where a > 2."""
        result = loaded_engine.query("a > 2")
        assert result.solver_status == "success"
        assert len(result.data) == 3

    def test_query_no_data_raises(self, engine) -> None:
        """query raises DataNotLoadedError when no data loaded."""
        from sidekick.data_processing.core import (
            DataNotLoadedError,
        )

        with pytest.raises(DataNotLoadedError):
            engine.query("a > 1")


# ---------------------------------------------------------------------------
# DataProcessorEngine — rename_column / drop_columns
# ---------------------------------------------------------------------------


class TestRenameAndDrop:
    """Tests for rename_column and drop_columns."""

    def test_rename_column_succeeds(self, loaded_engine) -> None:
        """rename_column changes the column name."""
        result = loaded_engine.rename_column("a", "alpha")
        assert result.solver_status == "success"
        assert "alpha" in loaded_engine.get_column_names()
        assert "a" not in loaded_engine.get_column_names()

    def test_rename_missing_column_raises(self, loaded_engine) -> None:
        """rename_column raises ColumnNotFoundError for unknown column."""
        from sidekick.data_processing.core import (
            ColumnNotFoundError,
        )

        with pytest.raises(ColumnNotFoundError):
            loaded_engine.rename_column("z", "zeta")

    def test_drop_columns_removes_them(self, loaded_engine) -> None:
        """drop_columns removes specified columns."""
        result = loaded_engine.drop_columns(["b"])
        assert result.solver_status == "success"
        assert "b" not in loaded_engine.get_column_names()

    def test_drop_missing_column_raises(self, loaded_engine) -> None:
        """drop_columns raises ColumnNotFoundError for unknown column."""
        from sidekick.data_processing.core import (
            ColumnNotFoundError,
        )

        with pytest.raises(ColumnNotFoundError):
            loaded_engine.drop_columns(["z"])


# ---------------------------------------------------------------------------
# DataProcessorEngine — add_calculated_column / transform_column
# ---------------------------------------------------------------------------


class TestCalculatedAndTransform:
    """Tests for add_calculated_column and transform_column."""

    def test_add_calculated_column_creates_new_column(self, loaded_engine) -> None:
        """add_calculated_column creates a new column from an expression."""
        result = loaded_engine.add_calculated_column("c", "a * 2")
        assert result.solver_status == "success"
        assert "c" in loaded_engine.get_column_names()

    def test_transform_column_log(self, loaded_engine) -> None:
        """transform_column 'log' applies natural log to column values."""
        result = loaded_engine.transform_column("a", "log")
        assert result.solver_status == "success"

    def test_transform_column_missing_column_raises(self, loaded_engine) -> None:
        """transform_column raises ColumnNotFoundError for unknown column."""
        from sidekick.data_processing.core import (
            ColumnNotFoundError,
        )

        with pytest.raises(ColumnNotFoundError):
            loaded_engine.transform_column("z", "log")


# ---------------------------------------------------------------------------
# DataProcessorEngine — fit_curve
# ---------------------------------------------------------------------------


class TestFitCurve:
    """Tests for DataProcessorEngine.fit_curve."""

    def test_linear_fit_perfect_data(self, loaded_engine) -> None:
        """Linear fit on perfectly linear data gives r_squared=1.0."""
        from sidekick.data_processing.core import FitType

        fr = loaded_engine.fit_curve("a", "b", FitType.LINEAR)
        assert abs(fr.r_squared - 1.0) < 1e-6

    def test_fit_result_has_coefficients(self, loaded_engine) -> None:
        """FitResult from linear fit contains non-empty coefficients."""
        from sidekick.data_processing.core import FitType

        fr = loaded_engine.fit_curve("a", "b", FitType.LINEAR)
        assert fr.theta_optimal is not None
        assert len(fr.theta_optimal) > 0

    def test_polynomial_fit(self, loaded_engine) -> None:
        """Polynomial fit returns a FitResult with r_squared."""
        from sidekick.data_processing.core import FitType

        fr = loaded_engine.fit_curve("a", "b", FitType.POLYNOMIAL, degree=2)
        assert fr.r_squared is not None

    def test_fit_missing_column_raises(self, loaded_engine) -> None:
        """fit_curve raises ColumnNotFoundError for unknown x column."""
        from sidekick.data_processing.core import (
            ColumnNotFoundError,
            FitType,
        )

        with pytest.raises(ColumnNotFoundError):
            loaded_engine.fit_curve("z", "b", FitType.LINEAR)


# ---------------------------------------------------------------------------
# DataProcessorEngine — smooth_column
# ---------------------------------------------------------------------------


class TestSmoothColumn:
    """Tests for DataProcessorEngine.smooth_column."""

    def _long_engine(self) -> DataProcessorEngine:
        """Engine with 20-row data for smoothing."""
        from sidekick.data_processing.core import (
            DataProcessorEngine,
        )

        e = DataProcessorEngine()
        e.load_dataframe(
            pd.DataFrame(
                {"a": [float(i) for i in range(20)], "b": [float(i) for i in range(20)]}
            )
        )
        return e

    def test_moving_average_succeeds(self) -> None:
        """smooth_column with 'moving_average' returns success."""
        e = self._long_engine()
        result = e.smooth_column("a", "moving_average", window=5)
        assert result.solver_status == "success"

    def test_median_smooth_succeeds(self) -> None:
        """smooth_column with 'median' returns success."""
        e = self._long_engine()
        result = e.smooth_column("a", "median", kernel=5)
        assert result.solver_status == "success"

    def test_data_processing_core_extended_unknown_method_raises(self) -> None:
        """smooth_column with unknown method raises UnsupportedOperationError."""
        from sidekick.data_processing.core import (
            UnsupportedOperationError,
        )

        e = self._long_engine()
        with pytest.raises(UnsupportedOperationError):
            e.smooth_column("a", "unknown_method")

    def test_smooth_no_data_raises(self, engine) -> None:
        """smooth_column raises DataNotLoadedError when no data loaded."""
        from sidekick.data_processing.core import (
            DataNotLoadedError,
        )

        with pytest.raises(DataNotLoadedError):
            engine.smooth_column("a", "moving_average")


# ---------------------------------------------------------------------------
# DataProcessorEngine — reset and undo
# ---------------------------------------------------------------------------


class TestResetAndUndo:
    """Tests for DataProcessorEngine.reset and _undo."""

    def test_reset_restores_original_data(self, loaded_engine) -> None:
        """reset() restores the DataFrame to its original state."""
        loaded_engine.drop_columns(["b"])
        result = loaded_engine.reset()
        # reset restores from original — data is present again
        assert result.solver_status == "success"
        assert loaded_engine.has_data() is True

    def test_undo_reverses_rename(self, loaded_engine) -> None:
        """_undo() reverses a rename_column operation."""
        loaded_engine.rename_column("a", "alpha")
        assert "alpha" in loaded_engine.get_column_names()
        loaded_engine._undo()
        assert "a" in loaded_engine.get_column_names()
