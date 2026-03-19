"""Tests for src.shared.python.upstream_drift_tools.data_processing.core (Issues #1949, #1744)."""

from __future__ import annotations

import pandas as pd

from src.shared.python.upstream_drift_tools.data_processing.core import (
    AggregationType,
    ColumnStats,
    DataFormat,
    DataProcessorEngine,
    FitType,
    ProcessingResult,
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
    def test_construct(self) -> None:
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
        assert result.success is True
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
        assert result.success is True

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
