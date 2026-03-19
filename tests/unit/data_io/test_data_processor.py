"""Tests for src.shared.python.data_processing.processor (Issues #1949, #1744)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.shared.python.data_processing.processor import DataProcessor, DatasetInfo

# ---------------------------------------------------------------------------
# DatasetInfo dataclass
# ---------------------------------------------------------------------------


class TestDatasetInfo:
    def test_defaults(self) -> None:
        info = DatasetInfo()
        assert info.name == ""
        assert info.num_rows == 0
        assert info.columns == []

    def test_stores_values(self) -> None:
        info = DatasetInfo(name="test", num_rows=10, num_columns=3)
        assert info.name == "test"
        assert info.num_rows == 10


# ---------------------------------------------------------------------------
# DataProcessor — construction and empty state
# ---------------------------------------------------------------------------


class TestDataProcessorEmpty:
    def test_dataframe_raises_when_empty(self) -> None:
        dp = DataProcessor()
        with pytest.raises(RuntimeError, match="No data loaded"):
            _ = dp.dataframe

    def test_info_returns_empty_info_when_nothing_loaded(self) -> None:
        dp = DataProcessor()
        info = dp.info
        assert isinstance(info, DatasetInfo)
        assert info.num_rows == 0

    def test_history_starts_empty(self) -> None:
        dp = DataProcessor()
        assert dp.history == []


# ---------------------------------------------------------------------------
# DataProcessor — load_dataframe
# ---------------------------------------------------------------------------


class TestDataProcessorLoadDataframe:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({"t": [0.0, 0.1, 0.2], "x": [1.0, 2.0, 3.0]})

    def test_load_dataframe_returns_self(self) -> None:
        dp = DataProcessor()
        result = dp.load_dataframe(self._make_df())
        assert result is dp

    def test_dataframe_accessible_after_load(self) -> None:
        dp = DataProcessor()
        df = self._make_df()
        dp.load_dataframe(df)
        assert len(dp.dataframe) == 3

    def test_info_populated_after_load(self) -> None:
        dp = DataProcessor()
        dp.load_dataframe(self._make_df(), name="test_data")
        info = dp.info
        assert info.num_rows == 3
        assert info.num_columns == 2

    def test_columns_recorded_in_info(self) -> None:
        dp = DataProcessor()
        dp.load_dataframe(self._make_df())
        assert "t" in dp.info.columns
        assert "x" in dp.info.columns


# ---------------------------------------------------------------------------
# DataProcessor — drop_columns / rename_columns
# ---------------------------------------------------------------------------


class TestDataProcessorColumnOps:
    def _loaded_dp(self) -> DataProcessor:
        dp = DataProcessor()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        dp.load_dataframe(df)
        return dp

    def test_drop_columns(self) -> None:
        dp = self._loaded_dp()
        dp.drop_columns(["c"])
        assert "c" not in dp.dataframe.columns
        assert "a" in dp.dataframe.columns

    def test_drop_columns_returns_self(self) -> None:
        dp = self._loaded_dp()
        result = dp.drop_columns(["c"])
        assert result is dp

    def test_rename_columns(self) -> None:
        dp = self._loaded_dp()
        dp.rename_columns({"a": "alpha"})
        assert "alpha" in dp.dataframe.columns
        assert "a" not in dp.dataframe.columns

    def test_rename_returns_self(self) -> None:
        dp = self._loaded_dp()
        result = dp.rename_columns({"b": "beta"})
        assert result is dp


# ---------------------------------------------------------------------------
# DataProcessor — sort / dropna
# ---------------------------------------------------------------------------


class TestDataProcessorSort:
    def test_sort_ascending(self) -> None:
        dp = DataProcessor()
        df = pd.DataFrame({"v": [3, 1, 2]})
        dp.load_dataframe(df)
        dp.sort("v", ascending=True)
        assert list(dp.dataframe["v"]) == [1, 2, 3]

    def test_sort_descending(self) -> None:
        dp = DataProcessor()
        df = pd.DataFrame({"v": [3, 1, 2]})
        dp.load_dataframe(df)
        dp.sort("v", ascending=False)
        assert list(dp.dataframe["v"]) == [3, 2, 1]

    def test_dropna_removes_nulls(self) -> None:
        import numpy as np

        dp = DataProcessor()
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        dp.load_dataframe(df)
        dp.dropna()
        assert len(dp.dataframe) == 2

    def test_dropna_returns_self(self) -> None:
        dp = DataProcessor()
        dp.load_dataframe(pd.DataFrame({"x": [1, 2]}))
        result = dp.dropna()
        assert result is dp


# ---------------------------------------------------------------------------
# DataProcessor — describe
# ---------------------------------------------------------------------------


class TestDataProcessorDescribe:
    def test_describe_returns_dict(self) -> None:
        dp = DataProcessor()
        dp.load_dataframe(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
        result = dp.describe()
        assert isinstance(result, dict)
