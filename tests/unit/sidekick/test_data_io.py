"""Tests for sidekick.data_processing.io (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sidekick.data_processing.io import (
    DataReader,
    DataWriter,
    FileFormatDetector,
)

# ---------------------------------------------------------------------------
# FileFormatDetector
# ---------------------------------------------------------------------------


class TestFileFormatDetector:
    def test_csv_detected(self) -> None:
        assert FileFormatDetector.detect_format(Path("data.csv")) == "csv"

    def test_tsv_detected(self) -> None:
        assert FileFormatDetector.detect_format(Path("data.tsv")) == "tsv"

    def test_json_detected(self) -> None:
        assert FileFormatDetector.detect_format(Path("data.json")) == "json"

    def test_excel_detected(self) -> None:
        assert FileFormatDetector.detect_format(Path("data.xlsx")) == "excel"

    def test_numpy_detected(self) -> None:
        assert FileFormatDetector.detect_format(Path("data.npy")) == "numpy"

    def test_pickle_detected(self) -> None:
        assert FileFormatDetector.detect_format(Path("data.pkl")) == "pickle"

    def test_unknown_extension_returns_none(self) -> None:
        result = FileFormatDetector.detect_format(Path("data.xyz"))
        assert result is None

    def test_data_io_case_insensitive(self) -> None:
        assert FileFormatDetector.detect_format(Path("data.CSV")) == "csv"

    def test_get_supported_extensions_nonempty(self) -> None:
        exts = FileFormatDetector.get_supported_extensions()
        assert len(exts) > 0
        assert ".csv" in exts


# ---------------------------------------------------------------------------
# DataReader
# ---------------------------------------------------------------------------


class TestDataReaderCSV:
    def test_read_csv_returns_dataframe(self, tmp_path: Path) -> None:
        p = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False)
        df = DataReader.read_file(p)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_read_csv_explicit_format(self, tmp_path: Path) -> None:
        p = tmp_path / "data.dat"
        pd.DataFrame({"x": [10]}).to_csv(p, index=False)
        df = DataReader.read_file(p, format_type="csv")
        assert "x" in df.columns

    def test_read_tsv(self, tmp_path: Path) -> None:
        p = tmp_path / "test.tsv"
        pd.DataFrame({"a": [5, 6]}).to_csv(p, sep="\t", index=False)
        df = DataReader.read_file(p)
        assert "a" in df.columns

    def test_read_json(self, tmp_path: Path) -> None:
        p = tmp_path / "test.json"
        pd.DataFrame({"v": [1, 2, 3]}).to_json(p, orient="records")
        df = DataReader.read_file(p)
        assert "v" in df.columns

    def test_read_numpy(self, tmp_path: Path) -> None:
        p = tmp_path / "test.npy"
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.save(str(p), arr)
        df = DataReader.read_file(p)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_read_pickle_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "data.pkl"
        p.write_text("dummy")
        with pytest.raises(
            ValueError, match="Pickle format is disabled for security reasons"
        ):
            DataReader.read_file(p)

    def test_data_io_unsupported_format_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "data.xyz"
        p.write_text("nothing")
        with pytest.raises(ValueError):
            DataReader.read_file(p)


# ---------------------------------------------------------------------------
# DataWriter
# ---------------------------------------------------------------------------


class TestDataWriterCSV:
    def test_write_csv_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "out.csv"
        df = pd.DataFrame({"x": [1, 2, 3]})
        DataWriter.write_file(df, p)
        assert p.exists()

    def test_write_read_csv_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "out.csv"
        df_orig = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        DataWriter.write_file(df_orig, p)
        df_read = DataReader.read_file(p)
        assert list(df_read.columns) == ["a", "b"]
        assert df_read["a"].tolist() == [1, 2]

    def test_write_tsv_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "out.tsv"
        df_orig = pd.DataFrame({"c": [10, 20]})
        DataWriter.write_file(df_orig, p)
        df_read = DataReader.read_file(p)
        assert df_read["c"].tolist() == [10, 20]

    def test_write_json_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        df_orig = pd.DataFrame({"z": [7, 8, 9]})
        DataWriter.write_file(df_orig, p)
        df_read = DataReader.read_file(p)
        assert sorted(df_read["z"].tolist()) == [7, 8, 9]

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "out.csv"
        DataWriter.write_file(pd.DataFrame({"v": [1]}), p)
        assert p.exists()

    def test_data_io_unsupported_format_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "data.xyz"
        with pytest.raises(ValueError):
            DataWriter.write_file(pd.DataFrame({"v": [1]}), p)
