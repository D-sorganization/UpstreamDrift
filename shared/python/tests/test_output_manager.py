"""Tests for the output manager."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from shared.python.output_manager import (
    OutputFormat,
    OutputManager,
    load_results,
    save_results,
)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output manager."""
    manager = OutputManager(base_path=tmp_path)
    manager.create_output_structure()
    return manager


def test_initialization(tmp_path):
    """Test directory creation."""
    manager = OutputManager(base_path=tmp_path)
    manager.create_output_structure()

    for dir_name in ["simulations", "analysis", "exports", "reports", "cache"]:
        assert (tmp_path / dir_name).exists()

    assert (tmp_path / "simulations" / "mujoco").exists()


def test_save_load_csv(temp_output_dir):
    """Test saving and loading CSV."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = temp_output_dir.save_simulation_results(
        df, "test_data", OutputFormat.CSV, engine="mujoco"
    )

    assert path.exists()
    assert path.suffix == ".csv"

    loaded = temp_output_dir.load_simulation_results(
        "test_data", OutputFormat.CSV, engine="mujoco"
    )
    pd.testing.assert_frame_equal(df, loaded)


def test_save_load_json(temp_output_dir):
    """Test saving and loading JSON."""
    # When saving raw list, it wraps it in {results: [...]}
    data = [1, 2, 3]
    path = temp_output_dir.save_simulation_results(
        data, "test_json", OutputFormat.JSON, engine="mujoco"
    )

    assert path.exists()
    assert path.suffix == ".json"

    loaded = temp_output_dir.load_simulation_results(
        "test_json", OutputFormat.JSON, engine="mujoco"
    )
    # load_simulation_results unwraps "results" key
    assert loaded == [1, 2, 3]

    # Test with existing structure
    data_struct = {"results": [4, 5], "other": "val"}
    temp_output_dir.save_simulation_results(
        data_struct, "test_json_struct", OutputFormat.JSON, engine="mujoco"
    )
    loaded2 = temp_output_dir.load_simulation_results(
        "test_json_struct", OutputFormat.JSON, engine="mujoco"
    )
    # If input had "results", and save wraps it again?
    # save_simulation_results puts input into "results" key of output dict.
    # So {results: {results: [4,5], other: val}, ...}
    # Then load unwraps outer results.
    assert loaded2 == data_struct


def test_save_load_parquet(temp_output_dir):
    """Test saving and loading Parquet."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pytest.skip("pyarrow not installed")

    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    try:
        path = temp_output_dir.save_simulation_results(
            df, "test_pq", OutputFormat.PARQUET, engine="mujoco"
        )
    except ImportError:
        pytest.skip("Parquet support missing")

    assert path.exists()
    assert path.suffix == ".parquet"

    loaded = temp_output_dir.load_simulation_results(
        "test_pq", OutputFormat.PARQUET, engine="mujoco"
    )
    pd.testing.assert_frame_equal(df, loaded)


def test_save_pickle_disabled(temp_output_dir):
    """Test that pickle is disabled for security."""
    data = [1, 2, 3]
    with pytest.raises(ValueError, match="Security: Pickle format is disabled"):
        temp_output_dir.save_simulation_results(data, "test", OutputFormat.PICKLE)


def test_export_report_html(temp_output_dir):
    """Test HTML report generation."""
    data = {"accuracy": 0.95, "speed": 100}
    path = temp_output_dir.export_analysis_report(data, "test_report", "html")

    assert path.exists()
    assert path.suffix == ".html"
    content = path.read_text()
    assert "accuracy" in content
    assert "0.95" in content


def test_cleanup_old_files(temp_output_dir):
    """Test file cleanup logic."""
    # Create an old file
    old_file = temp_output_dir.directories["simulations"] / "old.csv"
    old_file.touch()

    # Modify mtime to be 60 days ago
    old_time = (datetime.now() - timedelta(days=60)).timestamp()
    os.utime(old_file, (old_time, old_time))

    # Create a new file
    new_file = temp_output_dir.directories["simulations"] / "new.csv"
    new_file.touch()

    cleaned = temp_output_dir.cleanup_old_files(max_age_days=30)

    assert cleaned == 1
    # Old file should be moved to archive
    archive_file = temp_output_dir.base_path / "archive" / "simulations" / "old.csv"
    assert archive_file.exists()
    assert not old_file.exists()
    assert new_file.exists()


def test_convenience_functions(temp_output_dir, monkeypatch):
    """Test the top-level save_results and load_results functions."""
    # They create a new OutputManager internally, so we need to patch default path or rely on default
    # Monkeypatch OutputManager class used in the module

    # Actually, OutputManager default uses project root or cwd.
    # We can just verify they run, but better to control the path.
    # The convenience functions instantiate OutputManager() without args.

    # Let's mock OutputManager in shared.python.output_manager
    class MockManager:
        def __init__(self, base_path=None):
            pass

        def save_simulation_results(self, *args):
            return Path("mock_path.csv")

        def load_simulation_results(self, *args):
            return "mock_result"

    monkeypatch.setattr("shared.python.output_manager.OutputManager", MockManager)

    path = save_results([1, 2], "test")
    assert path == "mock_path.csv"

    res = load_results("test")
    assert res == "mock_result"
