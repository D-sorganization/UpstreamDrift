# ruff: noqa: E501
"""Comprehensive tests for the data_processing.processor module.

Covers DataProcessor and DatasetInfo to achieve 100% coverage.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from data_processing.processor import (
    SUPPORTED_FILTER_TYPES,
    DataProcessor,
    DatasetInfo,
)

from contracts import PreconditionError

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Simple numeric DataFrame with a time column."""
    return pd.DataFrame(
        {
            "time": np.linspace(0, 1, 100),
            "x": np.sin(np.linspace(0, 10, 100)),
            "y": np.cos(np.linspace(0, 10, 100)),
        }
    )


@pytest.fixture
def dp(sample_df: pd.DataFrame) -> DataProcessor:
    proc = DataProcessor()
    proc.load_dataframe(sample_df, name="test")
    return proc


# ──────────────────────────────────────────────────────────────────────────────
# DatasetInfo
# ──────────────────────────────────────────────────────────────────────────────


class TestDatasetInfo:
    def test_default_values(self):
        info = DatasetInfo()
        assert info.name == ""
        assert info.num_rows == 0
        assert info.columns == []
        assert info.dtypes == {}
        assert info.memory_mb == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# DataProcessor.load_dataframe
# ──────────────────────────────────────────────────────────────────────────────


class TestLoadDataframe:
    def test_load_dataframe_sets_df(self, sample_df: pd.DataFrame):
        dp = DataProcessor()
        dp.load_dataframe(sample_df, name="test")
        assert dp.dataframe is not None
        assert len(dp.dataframe) == 100

    def test_load_dataframe_records_history(self, sample_df: pd.DataFrame):
        dp = DataProcessor()
        dp.load_dataframe(sample_df, name="my_data")
        assert len(dp.history) == 1
        assert "my_data" in dp.history[0]

    def test_load_dataframe_invalid_type_raises(self):
        dp = DataProcessor()
        with pytest.raises(PreconditionError):
            dp.load_dataframe("not a dataframe")  # type: ignore[arg-type]

    def test_load_dataframe_empty_name_raises(self, sample_df: pd.DataFrame):
        dp = DataProcessor()
        with pytest.raises(PreconditionError):
            dp.load_dataframe(sample_df, name="")


# ──────────────────────────────────────────────────────────────────────────────
# DataProcessor.info and properties
# ──────────────────────────────────────────────────────────────────────────────


class TestInfoProperty:
    def test_info_without_data(self):
        dp = DataProcessor()
        info = dp.info
        assert info.num_rows == 0
        assert info.name == ""

    def test_info_with_data(self, dp: DataProcessor):
        info = dp.info
        assert info.num_rows == 100
        assert info.num_columns == 3
        assert "time" in info.columns
        assert info.memory_mb > 0

    def test_info_name_from_source_path(self, tmp_path: Path, sample_df: pd.DataFrame):
        csv_file = tmp_path / "my_data.csv"
        sample_df.to_csv(csv_file, index=False)
        proc = DataProcessor()
        proc.load(csv_file)
        assert proc.info.name == "my_data"

    def test_info_untitled_when_no_source_path(self, sample_df: pd.DataFrame):
        proc = DataProcessor()
        proc.load_dataframe(sample_df, name="x")
        # source_path is "" so name is "untitled"
        assert proc.info.name == "untitled"

    def test_dataframe_setter(self, sample_df: pd.DataFrame):
        dp = DataProcessor()
        dp.dataframe = sample_df
        assert len(dp.dataframe) == 100

    def test_dataframe_getter_raises_when_empty(self):
        dp = DataProcessor()
        with pytest.raises(RuntimeError, match="No data loaded"):
            _ = dp.dataframe

    def test_history_returns_copy(self, dp: DataProcessor):
        hist = dp.history
        hist.append("extra")
        assert len(dp.history) == 1  # original unmodified


# ──────────────────────────────────────────────────────────────────────────────
# DataProcessor.load (file formats)
# ──────────────────────────────────────────────────────────────────────────────


class TestLoadFile:
    def test_load_csv(self, tmp_path: Path, sample_df: pd.DataFrame):
        csv_file = tmp_path / "data.csv"
        sample_df.to_csv(csv_file, index=False)
        dp = DataProcessor()
        dp.load(csv_file)
        assert len(dp.dataframe) == 100

    def test_load_tsv(self, tmp_path: Path, sample_df: pd.DataFrame):
        tsv_file = tmp_path / "data.tsv"
        sample_df.to_csv(tsv_file, sep="\t", index=False)
        dp = DataProcessor()
        dp.load(tsv_file)
        assert len(dp.dataframe) == 100

    def test_load_txt(self, tmp_path: Path, sample_df: pd.DataFrame):
        txt_file = tmp_path / "data.txt"
        sample_df.to_csv(txt_file, index=False)
        dp = DataProcessor()
        dp.load(txt_file)
        assert len(dp.dataframe) == 100

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("pyarrow"),
        reason="pyarrow not installed",
    )
    def test_load_parquet(self, tmp_path: Path, sample_df: pd.DataFrame):
        pq_file = tmp_path / "data.parquet"
        sample_df.to_parquet(pq_file)
        dp = DataProcessor()
        dp.load(pq_file)
        assert len(dp.dataframe) == 100

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("openpyxl"),
        reason="openpyxl not installed",
    )
    def test_load_xlsx(self, tmp_path: Path, sample_df: pd.DataFrame):
        xlsx_file = tmp_path / "data.xlsx"
        sample_df.to_excel(xlsx_file, index=False)
        dp = DataProcessor()
        dp.load(xlsx_file)
        assert "time" in dp.dataframe.columns

    def test_load_unsupported_format_raises(self, tmp_path: Path):
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("abc")
        dp = DataProcessor()
        with pytest.raises(ValueError, match="Unsupported"):
            dp.load(bad_file)

    def test_load_dat_fallback(self, tmp_path: Path):
        """When dat_importer is unavailable, fallback to whitespace-delimited."""
        dat_file = tmp_path / "data.dat"
        # Write with header so pandas whitespace reader produces 2 data rows
        dat_file.write_text("a b\n1.0 2.0\n3.0 4.0\n")
        dp = DataProcessor()
        with patch.dict(
            "sys.modules",
            {
                "data_processor": None,
                "data_processor.core": None,
                "data_processor.core.dat_importer": None,
            },
        ):
            dp.load(dat_file)
        assert len(dp.dataframe) == 2

    def test_load_records_history(self, tmp_path: Path, sample_df: pd.DataFrame):
        csv_file = tmp_path / "data.csv"
        sample_df.to_csv(csv_file, index=False)
        dp = DataProcessor()
        dp.load(csv_file)
        assert len(dp.history) == 1
        assert "data" in dp.history[0].lower() or "row" in dp.history[0].lower()

    def test_load_returns_self(self, tmp_path: Path, sample_df: pd.DataFrame):
        csv_file = tmp_path / "data.csv"
        sample_df.to_csv(csv_file, index=False)
        dp = DataProcessor()
        result = dp.load(csv_file)
        assert result is dp


# ──────────────────────────────────────────────────────────────────────────────
# DataProcessor.trim_time
# ──────────────────────────────────────────────────────────────────────────────


class TestTrimTime:
    def test_trim_reduces_rows(self, dp: DataProcessor):
        original = len(dp.dataframe)
        dp.trim_time(0.2, 0.8)
        assert len(dp.dataframe) < original

    def test_trim_with_explicit_column(self, dp: DataProcessor):
        dp.trim_time(0.0, 0.5, time_column="time")
        assert dp.dataframe["time"].max() <= 0.5

    def test_trim_auto_detects_column(self, dp: DataProcessor):
        dp.trim_time(0.0, 0.5)
        assert dp.dataframe["time"].max() <= 0.5

    def test_trim_invalid_start_raises(self, dp: DataProcessor):
        with pytest.raises(PreconditionError):
            dp.trim_time("x", 0.5)  # type: ignore[arg-type]

    def test_trim_end_less_than_start_raises(self, dp: DataProcessor):
        with pytest.raises(PreconditionError):
            dp.trim_time(0.8, 0.2)

    def test_trim_records_history(self, dp: DataProcessor):
        dp.trim_time(0.2, 0.8)
        assert any("Trimmed" in h for h in dp.history)

    def test_trim_fallback_first_column_when_no_time(self):
        """_detect_time_column falls back to first column."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        proc = DataProcessor()
        proc.load_dataframe(df, name="test")
        proc.trim_time(1.5, 2.5)
        assert list(proc.dataframe["a"]) == [2.0]


# ──────────────────────────────────────────────────────────────────────────────
# DataProcessor.resample
# ──────────────────────────────────────────────────────────────────────────────


class TestResample:
    def test_resample_changes_length(self, dp: DataProcessor):
        """Force the pandas fallback path by patching away the real resample_data."""
        with patch.dict("sys.modules", {"data_processor.core.signal_processing": None}):
            dp.resample(target_rate=50.0)
        assert len(dp.dataframe) > 0

    def test_resample_invalid_rate_raises(self, dp: DataProcessor):
        with pytest.raises(PreconditionError):
            dp.resample(target_rate=-1.0)

    def test_resample_records_history(self, dp: DataProcessor):
        with patch.dict("sys.modules", {"data_processor.core.signal_processing": None}):
            dp.resample(target_rate=50.0)
        assert any("Resampled" in h for h in dp.history)

    def test_resample_with_explicit_time_column(self, dp: DataProcessor):
        with patch.dict("sys.modules", {"data_processor.core.signal_processing": None}):
            dp.resample(target_rate=50.0, time_column="time")
        assert len(dp.dataframe) > 0


# ──────────────────────────────────────────────────────────────────────────────
# DataProcessor.apply_filter
# ──────────────────────────────────────────────────────────────────────────────


class TestApplyFilter:
    def test_moving_average_filter(self, dp: DataProcessor):
        dp.apply_filter("moving_average", window_size=5)
        assert any("moving_average" in h for h in dp.history)

    def test_median_filter(self, dp: DataProcessor):
        dp.apply_filter("median", window_size=5)
        assert any("median" in h for h in dp.history)

    def test_savgol_filter(self, dp: DataProcessor):
        dp.apply_filter("savgol", window_size=11, order=3)
        assert any("savgol" in h for h in dp.history)

    def test_butterworth_filter(self, dp: DataProcessor):
        dp.apply_filter("butterworth", cutoff=10.0, order=2)
        assert any("butterworth" in h for h in dp.history)

    def test_filter_with_specific_columns(self, dp: DataProcessor):
        dp.apply_filter("moving_average", columns=["x"], window_size=5)
        assert any("moving_average" in h for h in dp.history)

    def test_unknown_filter_type_raises(self, dp: DataProcessor):
        with pytest.raises(ValueError, match="Unknown filter"):
            dp.apply_filter("nonexistent_filter")

    def test_zero_window_size_raises(self, dp: DataProcessor):
        with pytest.raises(ValueError, match="window_size"):
            dp.apply_filter("moving_average", window_size=0)

    def test_invalid_columns_raises(self, dp: DataProcessor):
        with pytest.raises(ValueError, match="No valid columns"):
            dp.apply_filter("moving_average", columns=["nonexistent_col"])

    def test_filter_fallback_without_scipy(self, dp: DataProcessor):
        """When scipy is unavailable, fallback to rolling mean covers remaining branch."""
        with patch(
            "data_processing.processor.DataProcessor._apply_filter_with_scipy",
            side_effect=ImportError("no scipy"),
        ):
            dp.apply_filter("moving_average", window_size=5)
        assert any("moving_average" in h for h in dp.history)

    def test_supported_filter_types_constant(self):
        assert "butterworth" in SUPPORTED_FILTER_TYPES
        assert "moving_average" in SUPPORTED_FILTER_TYPES
        assert "median" in SUPPORTED_FILTER_TYPES
        assert "savgol" in SUPPORTED_FILTER_TYPES


# ──────────────────────────────────────────────────────────────────────────────
# DataProcessor.apply_formula
# ──────────────────────────────────────────────────────────────────────────────


class TestApplyFormula:
    def test_apply_formula_creates_column(self, dp: DataProcessor):
        dp.apply_formula("speed", "x + y")
        assert "speed" in dp.dataframe.columns

    def test_apply_formula_records_history(self, dp: DataProcessor):
        dp.apply_formula("speed", "x + y")
        assert any("speed" in h for h in dp.history)

    def test_apply_formula_invalid_column_name_raises(self, dp: DataProcessor):
        with pytest.raises(PreconditionError):
            dp.apply_formula("", "x + y")

    def test_apply_formula_empty_expression_raises(self, dp: DataProcessor):
        with pytest.raises(PreconditionError):
            dp.apply_formula("result", "")

    # ------------------------------------------------------------------
    # Blocked-expression tests (issue #2481)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "expression",
        [
            # dunder attribute access
            "x.__class__",
            "x.__dict__",
            "x.__builtins__",
            # import injection
            "import os",
            "__import__('os')",
            # exec / eval injection
            "exec('print(1)')",
            "eval('1+1')",
            # file access
            "open('/etc/passwd')",
            # module attribute access
            "os.getcwd()",
            "sys.path",
            # subprocess execution
            "subprocess.call(['id'])",
            # lambda
            "lambda: None",
        ],
    )
    def test_apply_formula_blocks_dangerous_expression(
        self, dp: DataProcessor, expression: str
    ):
        """Dangerous patterns must be rejected before reaching pandas eval."""
        with pytest.raises(ValueError, match="forbidden pattern|Unsupported formula"):
            dp.apply_formula("result", expression)

    def test_apply_formula_numexpr_unavailable_raises(self, dp: DataProcessor):
        """If numexpr is not installed, eval must raise rather than silently fall back."""
        with (
            patch(
                "pandas.DataFrame.eval",
                side_effect=ImportError("numexpr not installed"),
            ),
            pytest.raises(RuntimeError, match="numexpr"),
        ):
            dp.apply_formula("result", "x + y")


# ──────────────────────────────────────────────────────────────────────────────
# DataProcessor.drop_columns / rename_columns / sort / dropna
# ──────────────────────────────────────────────────────────────────────────────


class TestTransformOps:
    def test_drop_columns(self, dp: DataProcessor):
        dp.drop_columns(["x"])
        assert "x" not in dp.dataframe.columns

    def test_drop_columns_nonexistent_is_ignored(self, dp: DataProcessor):
        dp.drop_columns(["nonexistent"])
        assert len(dp.dataframe.columns) == 3  # no change

    def test_drop_columns_invalid_raises(self, dp: DataProcessor):
        with pytest.raises(PreconditionError):
            dp.drop_columns([])

    def test_rename_columns(self, dp: DataProcessor):
        dp.rename_columns({"x": "x_renamed"})
        assert "x_renamed" in dp.dataframe.columns

    def test_rename_columns_empty_raises(self, dp: DataProcessor):
        with pytest.raises(PreconditionError):
            dp.rename_columns({})

    def test_sort_ascending(self, dp: DataProcessor):
        dp.sort("x", ascending=True)
        x_vals = dp.dataframe["x"].values
        assert all(x_vals[i] <= x_vals[i + 1] for i in range(len(x_vals) - 1))

    def test_sort_descending(self, dp: DataProcessor):
        dp.sort("x", ascending=False)
        x_vals = dp.dataframe["x"].values
        assert all(x_vals[i] >= x_vals[i + 1] for i in range(len(x_vals) - 1))

    def test_sort_invalid_column_raises(self, dp: DataProcessor):
        with pytest.raises(PreconditionError):
            dp.sort("")

    def test_sort_invalid_ascending_raises(self, dp: DataProcessor):
        with pytest.raises(PreconditionError):
            dp.sort("x", ascending="yes")  # type: ignore[arg-type]

    def test_dropna_no_columns(self, sample_df: pd.DataFrame):
        sample_df.loc[0, "x"] = float("nan")
        dp = DataProcessor()
        dp.load_dataframe(sample_df, name="test")
        dp.dropna()
        assert not dp.dataframe.isnull().any().any()

    def test_dropna_with_columns(self, sample_df: pd.DataFrame):
        sample_df.loc[0, "x"] = float("nan")
        dp = DataProcessor()
        dp.load_dataframe(sample_df, name="test")
        dp.dropna(columns=["x"])
        assert dp.dataframe["x"].isnull().sum() == 0


# ──────────────────────────────────────────────────────────────────────────────
# DataProcessor.describe / correlate / detect_outliers
# ──────────────────────────────────────────────────────────────────────────────


class TestAnalysis:
    def test_describe_returns_stats(self, dp: DataProcessor):
        result = dp.describe()
        assert "shape" in result
        assert "columns" in result
        assert "statistics" in result
        assert result["shape"] == [100, 3]

    def test_correlate_returns_dataframe(self, dp: DataProcessor):
        result = dp.correlate()
        assert isinstance(result, pd.DataFrame)
        assert "x" in result.columns
        assert "y" in result.columns

    def test_correlate_invalid_method_raises(self, dp: DataProcessor):
        with pytest.raises(PreconditionError):
            dp.correlate(method="")

    def test_detect_outliers_returns_mask(self, dp: DataProcessor):
        mask = dp.detect_outliers()
        assert isinstance(mask, pd.DataFrame)
        assert len(mask) > 0

    def test_detect_outliers_with_explicit_columns(self, dp: DataProcessor):
        mask = dp.detect_outliers(columns=["x"])
        assert "x" in mask.columns

    def test_detect_outliers_with_threshold(self, dp: DataProcessor):
        mask = dp.detect_outliers(threshold=2.0)
        assert isinstance(mask, pd.DataFrame)


# ──────────────────────────────────────────────────────────────────────────────
# DataProcessor.export
# ──────────────────────────────────────────────────────────────────────────────


class TestExport:
    def test_export_csv(self, dp: DataProcessor, tmp_path: Path):
        out = tmp_path / "out.csv"
        result = dp.export(out)
        assert result == out
        assert out.exists()

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("pyarrow"),
        reason="pyarrow not installed",
    )
    def test_export_parquet(self, dp: DataProcessor, tmp_path: Path):
        out = tmp_path / "out.parquet"
        dp.export(out)
        assert out.exists()

    def test_export_json(self, dp: DataProcessor, tmp_path: Path):
        out = tmp_path / "out.json"
        dp.export(out)
        assert out.exists()

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("openpyxl"),
        reason="openpyxl not installed",
    )
    def test_export_xlsx(self, dp: DataProcessor, tmp_path: Path):
        out = tmp_path / "out.xlsx"
        dp.export(out)
        assert out.exists()

    def test_export_unsupported_format_raises(self, dp: DataProcessor, tmp_path: Path):
        out = tmp_path / "out.xyz"
        with pytest.raises(ValueError, match="Unsupported"):
            dp.export(out)

    def test_export_records_history(self, dp: DataProcessor, tmp_path: Path):
        out = tmp_path / "out.csv"
        dp.export(out)
        assert any("Exported" in h for h in dp.history)


# ──────────────────────────────────────────────────────────────────────────────
# Method chaining
# ──────────────────────────────────────────────────────────────────────────────


class TestMethodChaining:
    def test_full_chain(self, tmp_path: Path, sample_df: pd.DataFrame):
        csv_file = tmp_path / "data.csv"
        sample_df.to_csv(csv_file, index=False)
        out_file = tmp_path / "out.csv"

        result_path = (
            DataProcessor()
            .load(csv_file)
            .trim_time(0.2, 0.8)
            .apply_filter("moving_average", window_size=5)
            .apply_formula("speed", "x + y")
            .drop_columns(["speed"])
            .sort("time")
            .dropna()
            .export(out_file)
        )
        assert result_path.exists()
