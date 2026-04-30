"""TDD Tests for Data Explorer Application.

Tests the data explorer's dataset discovery, loading, and format
validation capabilities.

Tests:
    1. Dataset Discovery — find files by supported extensions
    2. Dataset Loading — load and validate file metadata
    3. Format Validation — reject unsupported formats
    4. CSV/JSON parsing — extract column headers and keys
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.data_explorer.data_explorer_app import (
    SUPPORTED_EXTENSIONS,
    discover_datasets,
    load_dataset,
    main,
)


class TestSupportedFormats:
    """Test supported format definitions."""

    def test_csv_supported(self) -> None:
        """CSV is a supported format."""
        assert ".csv" in SUPPORTED_EXTENSIONS

    def test_json_supported(self) -> None:
        """JSON is a supported format."""
        assert ".json" in SUPPORTED_EXTENSIONS

    def test_hdf5_supported(self) -> None:
        """HDF5 is a supported format."""
        assert ".hdf5" in SUPPORTED_EXTENSIONS

    def test_c3d_supported(self) -> None:
        """C3D (motion capture) is a supported format."""
        assert ".c3d" in SUPPORTED_EXTENSIONS


class TestDatasetDiscovery:
    """Test dataset discovery in a directory."""

    def test_discover_csv_files(self, tmp_path: Path) -> None:
        """Discovers CSV files in directory."""
        (tmp_path / "data.csv").write_text("time,x,y\n1,2,3\n")
        (tmp_path / "ignore.txt").write_text("not a dataset")
        datasets = discover_datasets(tmp_path)
        assert len(datasets) == 1
        assert datasets[0].name == "data.csv"

    def test_discover_multiple_formats(self, tmp_path: Path) -> None:
        """Discovers files across multiple formats."""
        (tmp_path / "run1.csv").write_text("t,v\n")
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "readme.txt").write_text("ignore")
        datasets = discover_datasets(tmp_path)
        assert len(datasets) == 2

    def test_discover_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        datasets = discover_datasets(tmp_path)
        assert datasets == []

    def test_discover_nonexistent_dir_raises(self) -> None:
        """Non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            discover_datasets(Path("/nonexistent/path"))

    def test_discover_recursive(self, tmp_path: Path) -> None:
        """Discovers files in subdirectories."""
        subdir = tmp_path / "run_001"
        subdir.mkdir()
        (subdir / "results.csv").write_text("t,x\n")
        datasets = discover_datasets(tmp_path)
        assert len(datasets) == 1

    def test_results_sorted_by_name(self, tmp_path: Path) -> None:
        """Results are sorted alphabetically."""
        (tmp_path / "z_data.csv").write_text("t\n")
        (tmp_path / "a_data.csv").write_text("t\n")
        datasets = discover_datasets(tmp_path)
        assert datasets[0].name == "a_data.csv"
        assert datasets[1].name == "z_data.csv"


class TestDatasetLoading:
    """Test loading individual datasets."""

    def test_load_csv_metadata(self, tmp_path: Path) -> None:
        """Loading a CSV returns format and size metadata."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("time,position,velocity\n1,0.5,1.2\n")
        info = load_dataset(csv_file)
        assert info["format"] == "csv"
        assert info["size_bytes"] > 0
        assert info["columns"] == ["time", "position", "velocity"]

    def test_load_json_metadata(self, tmp_path: Path) -> None:
        """Loading a JSON returns top-level keys."""
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps({"engine": "mujoco", "dt": 0.001}))
        info = load_dataset(json_file)
        assert info["format"] == "json"
        assert "engine" in info["columns"]
        assert "dt" in info["columns"]

    def test_load_nonexistent_raises(self) -> None:
        """Loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset(Path("/nonexistent/data.csv"))

    def test_load_unsupported_format_raises(self, tmp_path: Path) -> None:
        """Loading an unsupported format raises ValueError."""
        bad_file = tmp_path / "data.xlsx"
        bad_file.write_text("binary")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_dataset(bad_file)


class TestMain:
    """Test the main() entry point."""

    def test_main_returns_zero(self) -> None:
        """Main returns 0 (success) when called."""
        result = main()
        assert result == 0
