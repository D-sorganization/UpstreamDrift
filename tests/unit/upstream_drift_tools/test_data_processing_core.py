"""Tests for src.shared.python.upstream_drift_tools.data_processing.core (Issues #1949, #1744, #2065)."""

from __future__ import annotations

import pandas as pd
import pytest
from src.shared.python.upstream_drift_tools.data_processing.core import (
    AggregationType,
    ColumnStats,
    DataFormat,
    DataProcessorEngine,
    FitType,
    ProcessingResult,
    _validate_dataframe_expression,
)
from src.shared.python.upstream_drift_tools.data_processing.exceptions import (
    TransformationError,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestDataFormat:
    def test_csv_value(self) -> None:
        assert DataFormat.CSV.value == "csv"

    def test_json_value(self) -> None:
        assert DataFormat.JSON.value == "json"


class TestAggregationType:
    def test_sum_value(self) -> None:
        assert AggregationType.SUM.value == "sum"

    def test_mean_value(self) -> None:
        assert AggregationType.MEAN.value == "mean"


class TestFitType:
    def test_linear_value(self) -> None:
        assert FitType.LINEAR.value == "linear"

    def test_polynomial_value(self) -> None:
        assert FitType.POLYNOMIAL.value == "polynomial"


# ---------------------------------------------------------------------------
# ColumnStats dataclass
# ---------------------------------------------------------------------------


class TestColumnStats:
    def test_data_processing_core_construct(self) -> None:
        stats = ColumnStats(
            name="x",
            dtype="float64",
            count=10,
            null_count=0,
            unique_count=10,
        )
        assert stats.name == "x"
        assert stats.count == 10

    def test_optional_fields_default_none(self) -> None:
        stats = ColumnStats(
            name="x",
            dtype="float64",
            count=5,
            null_count=0,
            unique_count=5,
        )
        assert stats.mean is None
        assert stats.std is None


# ---------------------------------------------------------------------------
# ProcessingResult dataclass
# ---------------------------------------------------------------------------


class TestProcessingResult:
    def test_success(self) -> None:
        result = ProcessingResult(success=True, message="ok")
        assert result.solver_status == "success"
        assert result.message == "ok"

    def test_default_stats_empty(self) -> None:
        result = ProcessingResult(success=False, message="err")
        assert result.stats == {}

    def test_timestamp_is_string(self) -> None:
        result = ProcessingResult(success=True, message="x")
        assert isinstance(result.timestamp, str)


# ---------------------------------------------------------------------------
# DataProcessorEngine
# ---------------------------------------------------------------------------


class TestDataProcessorEngineInit:
    def test_data_is_none(self) -> None:
        engine = DataProcessorEngine()
        assert engine.data is None

    def test_history_empty(self) -> None:
        engine = DataProcessorEngine()
        assert engine.history == []


class TestDataProcessorEngineLoad:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

    def test_load_stores_data(self) -> None:
        engine = DataProcessorEngine()
        df = self._make_df()
        engine.load_dataframe(df)
        assert engine.data is not None
        assert len(engine.data) == 3

    def test_load_returns_processing_result(self) -> None:
        engine = DataProcessorEngine()
        result = engine.load_dataframe(self._make_df())
        assert isinstance(result, ProcessingResult)
        assert result.solver_status == "success"

    def test_load_sets_original_data(self) -> None:
        engine = DataProcessorEngine()
        engine.load_dataframe(self._make_df())
        assert engine.original_data is not None

    def test_get_statistics_after_load(self) -> None:
        engine = DataProcessorEngine()
        engine.load_dataframe(self._make_df())
        stats = engine.get_statistics()
        assert isinstance(stats, dict)
        assert "a" in stats
        assert isinstance(stats["a"], ColumnStats)

    def test_no_data_returns_empty_stats(self) -> None:
        engine = DataProcessorEngine()
        result = engine.get_statistics()
        assert result == {}


# ---------------------------------------------------------------------------
# Expression validation (security -- issue #2065)
# ---------------------------------------------------------------------------


class TestValidateDataframeExpression:
    """Unit tests for the _validate_dataframe_expression guard in core."""

    def test_safe_arithmetic_passes(self) -> None:
        _validate_dataframe_expression("col_a + col_b")

    def test_safe_comparison_passes(self) -> None:
        _validate_dataframe_expression("velocity > 0")

    def test_import_rejected(self) -> None:
        with pytest.raises(ValueError, match="forbidden name"):
            _validate_dataframe_expression("__import__('subprocess')")

    def test_exec_rejected(self) -> None:
        with pytest.raises(ValueError, match="forbidden name"):
            _validate_dataframe_expression("exec('rm -rf /')")

    def test_eval_rejected(self) -> None:
        with pytest.raises(ValueError, match="forbidden name"):
            _validate_dataframe_expression("eval('2+2')")

    def test_dunder_attribute_rejected(self) -> None:
        with pytest.raises(ValueError, match="dunder"):
            _validate_dataframe_expression("x.__globals__")

    def test_lambda_rejected(self) -> None:
        with pytest.raises(ValueError, match="Disallowed"):
            _validate_dataframe_expression("lambda: None")

    def test_open_rejected(self) -> None:
        with pytest.raises(ValueError, match="forbidden name"):
            _validate_dataframe_expression("open('/etc/shadow')")

    def test_data_processing_core_syntax_error_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Syntax error"):
            _validate_dataframe_expression("=== bad")


class TestDataProcessorEngineAddCalculatedColumn:
    """Integration tests for add_calculated_column security (issue #2065)."""

    def _loaded_engine(self) -> DataProcessorEngine:
        engine = DataProcessorEngine()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        engine.load_dataframe(df)
        return engine

    def test_safe_expression_adds_column(self) -> None:
        engine = self._loaded_engine()
        result = engine.add_calculated_column("z", "x + y")
        assert result.solver_status == "success"
        assert engine.data is not None
        assert "z" in engine.data.columns
        assert list(engine.data["z"]) == [11.0, 22.0, 33.0]

    def test_exec_raises_transformation_error(self) -> None:
        engine = self._loaded_engine()
        with pytest.raises(TransformationError, match="forbidden name"):
            engine.add_calculated_column("z", "exec('import os')")

    def test_dunder_raises_transformation_error(self) -> None:
        engine = self._loaded_engine()
        with pytest.raises(TransformationError, match="dunder"):
            engine.add_calculated_column("z", "x.__class__")
