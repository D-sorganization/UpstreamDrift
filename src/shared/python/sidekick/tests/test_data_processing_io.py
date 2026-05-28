"""Tests for data_processing.io - DataReader, DataWriter, FileFormatDetector.

Covers all supported formats using tempfiles:
- CSV, TSV, JSON, Numpy, SQLite, Pickle
- FileFormatDetector extension mapping
- Error paths: unsupported format, parquet/matlab dependency checks
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small DataFrame used by all write/read round-trip tests."""
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


# ---------------------------------------------------------------------------
# DataReader
# ---------------------------------------------------------------------------


class TestDataReader:
    """Test DataReader.read_file() for each supported format."""

    def test_read_csv(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            sample_df.to_csv(f.name, index=False)
            result = DataReader.read_file(f.name)
        assert list(result.columns) == ["x", "y"]
        assert len(result) == 3

    def test_read_csv_with_explicit_format(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".data", delete=False) as f:
            sample_df.to_csv(f.name, index=False)
            result = DataReader.read_file(f.name, format_type="csv")
        assert list(result.columns) == ["x", "y"]

    def test_read_tsv(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
            sample_df.to_csv(f.name, sep="\t", index=False)
            result = DataReader.read_file(f.name)
        assert list(result.columns) == ["x", "y"]

    def test_read_json(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            sample_df.to_json(f.name, orient="records")
            result = DataReader.read_file(f.name)
        assert "x" in result.columns

    def test_read_numpy(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            arr = sample_df.values
            np.save(f.name, arr)
            result = DataReader.read_file(f.name)
        assert result.shape == (3, 2)

    def test_read_pickle(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            sample_df.to_pickle(f.name)
            with pytest.raises(ValueError, match="Pickle format is disabled"):
                DataReader.read_file(f.name, format_type="pickle")

    def test_read_sqlite(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataReader

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            conn = sqlite3.connect(f.name)
            sample_df.to_sql("data", conn, if_exists="replace", index=False)
            conn.close()
            result = DataReader.read_file(f.name)
        assert list(result.columns) == ["x", "y"]

    def test_read_unsupported_format_raises(self):
        from upstream_drift_tools.data_processing.io import DataReader

        with pytest.raises(ValueError, match="Unsupported or undetected"):
            DataReader.read_file("/tmp/file.xyz_unknown_ext")  # nosec B108

    def test_read_parquet_no_pyarrow_raises(self):
        """If PYARROW_AVAILABLE is False, reading parquet should raise ImportError."""
        import upstream_drift_tools.data_processing.io as io_mod
        from upstream_drift_tools.data_processing.io import DataReader

        original = io_mod.PYARROW_AVAILABLE
        try:
            io_mod.PYARROW_AVAILABLE = False
            with pytest.raises(ImportError, match="PyArrow"):
                DataReader.read_file("/tmp/fake.parquet")  # nosec B108
        finally:
            io_mod.PYARROW_AVAILABLE = original


# ---------------------------------------------------------------------------
# DataWriter
# ---------------------------------------------------------------------------


class TestDataWriter:
    """Test DataWriter.write_file() for each supported format."""

    def test_write_csv(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            DataWriter.write_file(sample_df, f.name)
            result = pd.read_csv(f.name)
        assert list(result.columns) == ["x", "y"]

    def test_write_tsv(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
            DataWriter.write_file(sample_df, f.name)
            result = pd.read_csv(f.name, sep="\t")
        assert list(result.columns) == ["x", "y"]

    def test_write_json(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            DataWriter.write_file(sample_df, f.name)
            result = pd.read_json(f.name)
        assert "x" in result.columns

    def test_write_pickle(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:  # noqa: SIM117
            with pytest.raises(ValueError, match="Pickle format is disabled"):
                DataWriter.write_file(sample_df, f.name, format_type="pickle")

    def test_write_numpy(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            DataWriter.write_file(sample_df, f.name)
            arr = np.load(f.name, allow_pickle=False)
        assert arr.shape == (3, 2)

    def test_write_sqlite(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            DataWriter.write_file(sample_df, f.name)
            conn = sqlite3.connect(f.name)
            result = pd.read_sql("SELECT * FROM data", conn)
            conn.close()
        assert list(result.columns) == ["x", "y"]

    def test_write_unsupported_format_raises(self, sample_df: pd.DataFrame):
        from upstream_drift_tools.data_processing.io import DataWriter

        with pytest.raises(ValueError, match="Unsupported or undetected"):
            DataWriter.write_file(sample_df, "output.xyz_unknown_ext")

    def test_write_creates_parent_dirs(self, sample_df: pd.DataFrame, tmp_path: Path):
        from upstream_drift_tools.data_processing.io import DataWriter

        nested = tmp_path / "a" / "b" / "c" / "out.csv"
        DataWriter.write_file(sample_df, nested)
        assert nested.exists()

    def test_write_parquet_no_pyarrow_raises(self, sample_df: pd.DataFrame):
        import upstream_drift_tools.data_processing.io as io_mod
        from upstream_drift_tools.data_processing.io import DataWriter

        original = io_mod.PYARROW_AVAILABLE
        try:
            io_mod.PYARROW_AVAILABLE = False
            with pytest.raises(ImportError, match="PyArrow"):
                DataWriter.write_file(sample_df, "/tmp/fake.parquet")  # nosec B108
        finally:
            io_mod.PYARROW_AVAILABLE = original


# ---------------------------------------------------------------------------
# FileFormatDetector
# ---------------------------------------------------------------------------


class TestFileFormatDetector:
    """Cover detect_format and get_supported_extensions."""

    @pytest.mark.parametrize(
        "ext,expected",
        [
            (".csv", "csv"),
            (".tsv", "tsv"),
            (".txt", "tsv"),
            (".xlsx", "excel"),
            (".xls", "excel"),
            (".json", "json"),
            (".npy", "numpy"),
            (".mat", "matlab"),
            (".db", "sqlite"),
            (".sqlite", "sqlite"),
        ],
    )
    def test_detect_known_extensions(self, ext: str, expected: str):
        from upstream_drift_tools.data_processing.io import FileFormatDetector

        result = FileFormatDetector.detect_format(f"/data/file{ext}")
        assert result == expected

    def test_detect_unknown_extension_returns_none(self):
        from upstream_drift_tools.data_processing.io import FileFormatDetector

        assert FileFormatDetector.detect_format("/data/file.xyz_unknown") is None

    def test_get_supported_extensions_returns_list(self):
        from upstream_drift_tools.data_processing.io import FileFormatDetector

        extensions = FileFormatDetector.get_supported_extensions()
        assert isinstance(extensions, list)
        assert ".csv" in extensions
        assert ".json" in extensions


# ---------------------------------------------------------------------------
# Additional format paths: Excel, Parquet (available), NumPy dict, MATLAB
# ---------------------------------------------------------------------------


try:
    import pyarrow  # noqa: F401

    _PYARROW_HERE = True
except ImportError:
    _PYARROW_HERE = False

requires_pyarrow = pytest.mark.skipif(not _PYARROW_HERE, reason="pyarrow not installed")


try:
    import openpyxl  # noqa: F401

    _OPENPYXL_HERE = True
except ImportError:
    _OPENPYXL_HERE = False

requires_openpyxl = pytest.mark.skipif(
    not _OPENPYXL_HERE, reason="openpyxl not installed"
)


class TestDataReaderAdditionalFormats:
    @requires_openpyxl
    def test_read_excel(self, sample_df: pd.DataFrame, tmp_path: Path):
        """Line 53: excel read path via read_file."""
        from upstream_drift_tools.data_processing.io import DataReader

        excel_path = tmp_path / "test.xlsx"
        sample_df.to_excel(excel_path, index=False)
        result = DataReader.read_file(excel_path)
        assert list(result.columns) == ["x", "y"]

    @requires_pyarrow
    def test_read_parquet_with_pyarrow_available(
        self, sample_df: pd.DataFrame, tmp_path: Path
    ):
        """Line 57: parquet read path when PYARROW_AVAILABLE=True."""
        from upstream_drift_tools.data_processing.io import DataReader

        pq_path = tmp_path / "test.parquet"
        sample_df.to_parquet(pq_path, index=False)
        result = DataReader.read_file(pq_path)
        assert list(result.columns) == ["x", "y"]

    def test_read_numpy_dict_format(self, tmp_path: Path):
        """Line 66: np.load returning non-ndarray (e.g. NpzFile) → data.item() path."""
        from unittest.mock import MagicMock, patch

        from upstream_drift_tools.data_processing.io import DataReader

        npy_path = tmp_path / "data.npy"
        npy_path.write_bytes(b"placeholder")

        # Simulate np.load returning something that is NOT an ndarray
        # (e.g. NpzFile), and whose .item() returns a dict
        data_dict = {"col_a": [1.0, 2.0], "col_b": [3.0, 4.0]}
        mock_result = MagicMock()  # not an np.ndarray instance
        mock_result.__class__ = object  # type: ignore[assignment]  # ensure isinstance(mock_result, np.ndarray) is False
        mock_result.item.return_value = data_dict

        # Patch np.load in the io module namespace
        with patch(
            "upstream_drift_tools.data_processing.io.np.load", return_value=mock_result
        ):
            result = DataReader.read_file(npy_path)
        assert "col_a" in result.columns

    def test_read_matlab_no_scipy_raises(self):
        """Lines 68-69: SCIPY_AVAILABLE=False raises ImportError for .mat files."""
        import upstream_drift_tools.data_processing.io as io_mod
        from upstream_drift_tools.data_processing.io import DataReader

        original = io_mod.SCIPY_AVAILABLE
        try:
            io_mod.SCIPY_AVAILABLE = False
            with pytest.raises(ImportError, match="SciPy"):
                DataReader.read_file("/tmp/fake.mat")  # nosec B108
        finally:
            io_mod.SCIPY_AVAILABLE = original

    def test_read_matlab_single_key(self, tmp_path: Path):
        """Lines 70-73: MATLAB file read with a single data key."""
        from unittest.mock import patch

        import upstream_drift_tools.data_processing.io as io_mod
        from upstream_drift_tools.data_processing.io import DataReader

        mat_path = tmp_path / "data.mat"
        mat_path.write_bytes(b"placeholder")

        mock_data = {"mydata": np.array([[1.0, 2.0], [3.0, 4.0]]), "__header__": b""}
        original = io_mod.SCIPY_AVAILABLE
        try:
            io_mod.SCIPY_AVAILABLE = True
            with patch("scipy.io.loadmat", return_value=mock_data):
                result = DataReader.read_file(mat_path)
        finally:
            io_mod.SCIPY_AVAILABLE = original
        assert result.shape == (2, 2)

    def test_read_matlab_multi_key(self, tmp_path: Path):
        """Lines 74-76: MATLAB file with multiple data keys → dict DataFrame."""
        from unittest.mock import patch

        import upstream_drift_tools.data_processing.io as io_mod
        from upstream_drift_tools.data_processing.io import DataReader

        mat_path = tmp_path / "data.mat"
        mat_path.write_bytes(b"placeholder")

        mock_data = {
            "x": np.array([1.0, 2.0, 3.0]),
            "y": np.array([4.0, 5.0, 6.0]),
            "__header__": b"",
        }
        original = io_mod.SCIPY_AVAILABLE
        try:
            io_mod.SCIPY_AVAILABLE = True
            with patch("scipy.io.loadmat", return_value=mock_data):
                result = DataReader.read_file(mat_path)
        finally:
            io_mod.SCIPY_AVAILABLE = original
        assert "x" in result.columns
        assert "y" in result.columns


class TestDataWriterAdditionalFormats:
    @requires_openpyxl
    def test_write_excel(self, sample_df: pd.DataFrame, tmp_path: Path):
        """Line 109: excel write path."""
        from upstream_drift_tools.data_processing.io import DataWriter

        excel_path = tmp_path / "out.xlsx"
        DataWriter.write_file(sample_df, excel_path)
        result = pd.read_excel(excel_path)
        assert list(result.columns) == ["x", "y"]

    @requires_pyarrow
    def test_write_parquet_with_pyarrow_available(
        self, sample_df: pd.DataFrame, tmp_path: Path
    ):
        """Line 113: parquet write when PYARROW_AVAILABLE=True."""
        from upstream_drift_tools.data_processing.io import DataWriter

        pq_path = tmp_path / "out.parquet"
        DataWriter.write_file(sample_df, pq_path)
        result = pd.read_parquet(pq_path)
        assert list(result.columns) == ["x", "y"]
