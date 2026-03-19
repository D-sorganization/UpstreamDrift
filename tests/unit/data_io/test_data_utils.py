"""Tests for src.shared.python.data_io.data_utils (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.shared.python.data_io.data_utils import (
    DataLoader,
    load_csv_data,
    load_json_data,
    load_numpy_data,
    save_csv_data,
    save_json_data,
    save_numpy_data,
)

# ---------------------------------------------------------------------------
# load_csv_data / save_csv_data
# ---------------------------------------------------------------------------


class TestCsvDataIO:
    def test_save_and_load_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = save_csv_data(df, p, index=False)
        assert result is True
        loaded = load_csv_data(p)
        assert loaded is not None
        assert len(loaded) == 3

    def test_load_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_csv_data(tmp_path / "missing.csv")

    def test_csv_columns_preserved(self, tmp_path: Path) -> None:
        p = tmp_path / "cols.csv"
        df = pd.DataFrame({"x": [1], "y": [2]})
        save_csv_data(df, p, index=False)
        loaded = load_csv_data(p)
        assert loaded is not None
        assert "x" in loaded.columns
        assert "y" in loaded.columns


# ---------------------------------------------------------------------------
# load_json_data / save_json_data
# ---------------------------------------------------------------------------


class TestJsonDataIO:
    def test_save_and_load_json(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        data = {"key": "value", "num": 42}
        result = save_json_data(data, p)
        assert result is True
        loaded = load_json_data(p)
        assert loaded == data

    def test_load_missing_json_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_json_data(tmp_path / "missing.json")

    def test_json_list_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "list.json"
        data = [1, 2, 3]
        save_json_data(data, p)
        loaded = load_json_data(p)
        assert loaded == data

    def test_json_indented(self, tmp_path: Path) -> None:
        p = tmp_path / "indented.json"
        save_json_data({"x": 1}, p, indent=4)
        content = p.read_text()
        # 4-space indented JSON has newlines
        assert "\n" in content


# ---------------------------------------------------------------------------
# load_numpy_data / save_numpy_data
# ---------------------------------------------------------------------------


class TestNumpyDataIO:
    def test_save_and_load_npy(self, tmp_path: Path) -> None:
        p = tmp_path / "arr.npy"
        arr = np.array([1.0, 2.0, 3.0])
        result = save_numpy_data(arr, p)
        assert result is True
        loaded = load_numpy_data(p)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, arr)

    def test_save_compressed_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "arr.npz"
        arr = np.ones((10, 10))
        result = save_numpy_data(arr, p, compressed=True)
        assert result is True
        # np.savez_compressed appends .npz if not already present
        npz_path = p if p.exists() else p.with_name(p.name + ".npz")
        assert npz_path.exists()

    def test_load_missing_npy_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_numpy_data(tmp_path / "missing.npy")


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_detect_csv_format(self, tmp_path: Path) -> None:
        p = tmp_path / "f.csv"
        loader = DataLoader(p)
        assert loader._format == "csv"

    def test_detect_json_format(self, tmp_path: Path) -> None:
        p = tmp_path / "f.json"
        loader = DataLoader(p)
        assert loader._format == "json"

    def test_detect_numpy_format(self, tmp_path: Path) -> None:
        p = tmp_path / "f.npy"
        loader = DataLoader(p)
        assert loader._format == "numpy"

    def test_load_csv_returns_dataframe(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        df = pd.DataFrame({"v": [1, 2]})
        df.to_csv(p, index=False)
        loader = DataLoader(p)
        result = loader.load()
        assert result is not None
        assert len(result) == 2

    def test_caching(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        df = pd.DataFrame({"v": [1, 2]})
        df.to_csv(p, index=False)
        loader = DataLoader(p)
        loader.load()  # prime cache
        # Overwrite file, but cached load should return old data
        pd.DataFrame({"v": [99]}).to_csv(p, index=False)
        cached = loader.load(use_cache=True)
        assert cached is not None
        assert list(cached["v"]) == [1, 2]
