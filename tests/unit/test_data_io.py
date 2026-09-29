"""Tests for sidekick.data_io (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

try:
    import pyarrow  # noqa: F401

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from sidekick.data_io import read_data, write_data

needs_parquet = pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


class TestWriteData:
    def test_write_csv(self, sample_df, tmp_path) -> None:
        path = tmp_path / "out.csv"
        result = write_data(sample_df, path)
        assert result == path
        assert path.exists()

    @needs_parquet
    def test_write_parquet(self, sample_df, tmp_path) -> None:
        path = tmp_path / "out.parquet"
        result = write_data(sample_df, path)
        assert result == path
        assert path.exists()

    @needs_parquet
    def test_write_csv_and_also_parquet(self, sample_df, tmp_path) -> None:
        # write_data writes to extension-specified format; also_csv only
        # applies when writing Parquet → also write CSV sibling
        path = tmp_path / "out.parquet"
        write_data(sample_df, path, also_csv=True)
        assert path.exists()
        assert (tmp_path / "out.csv").exists()

    def test_write_returns_path(self, sample_df, tmp_path) -> None:
        path = tmp_path / "data.csv"
        result = write_data(sample_df, path)
        assert isinstance(result, Path)

    def test_write_creates_parent_dirs(self, sample_df, tmp_path) -> None:
        path = tmp_path / "subdir" / "data.csv"
        write_data(sample_df, path)
        assert path.exists()

    def test_write_invalid_extension_raises(self, sample_df, tmp_path) -> None:
        path = tmp_path / "out.xlsx"
        with pytest.raises((ValueError, AssertionError)):
            write_data(sample_df, path)


class TestReadData:
    def test_read_csv(self, sample_df, tmp_path) -> None:
        path = tmp_path / "data.csv"
        write_data(sample_df, path)
        df = read_data(path)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 3

    @needs_parquet
    def test_read_parquet(self, sample_df, tmp_path) -> None:
        path = tmp_path / "data.parquet"
        write_data(sample_df, path)
        df = read_data(path)
        assert list(df.columns) == ["a", "b"]

    @needs_parquet
    def test_read_prefers_parquet_over_csv(self, sample_df, tmp_path) -> None:
        csv_path = tmp_path / "data.csv"
        parquet_path = tmp_path / "data.parquet"
        # Write both files
        csv_df = pd.DataFrame({"a": [99], "b": [0.0]})
        csv_df.to_csv(csv_path, index=False)
        sample_df.to_parquet(parquet_path, index=False)
        # read_data with CSV path and prefer_parquet=True should read parquet
        df = read_data(csv_path, prefer_parquet=True)
        assert len(df) == 3  # parquet has 3 rows

    def test_read_csv_when_no_parquet(self, sample_df, tmp_path) -> None:
        path = tmp_path / "data.csv"
        write_data(sample_df, path)
        df = read_data(path, prefer_parquet=True)
        assert len(df) == 3

    def test_roundtrip_csv(self, sample_df, tmp_path) -> None:
        path = tmp_path / "rt.csv"
        write_data(sample_df, path)
        df = read_data(path)
        pd.testing.assert_frame_equal(df, sample_df)

    @needs_parquet
    def test_roundtrip_parquet(self, sample_df, tmp_path) -> None:
        path = tmp_path / "rt.parquet"
        write_data(sample_df, path)
        df = read_data(path)
        pd.testing.assert_frame_equal(df, sample_df)

    def test_data_io_missing_file_raises(self, tmp_path) -> None:
        path = tmp_path / "nonexistent.csv"
        with pytest.raises((FileNotFoundError, OSError)):
            read_data(path)

    def test_invalid_extension_raises(self, tmp_path) -> None:
        path = tmp_path / "data.xlsx"
        with pytest.raises((ValueError, AssertionError)):
            read_data(path)
