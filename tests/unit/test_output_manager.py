import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from shared.python.output_manager import OutputFormat, OutputManager


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output testing."""
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir)


@pytest.fixture
def output_manager(temp_output_dir):
    """Create OutputManager instance with temporary directory."""
    return OutputManager(base_path=temp_output_dir)


def test_initialization(output_manager, temp_output_dir):
    """Test standard initialization."""
    assert output_manager.base_path == temp_output_dir
    assert isinstance(output_manager.directories, dict)
    assert "simulations" in output_manager.directories


def test_create_output_structure(output_manager):
    """Test directory structure creation."""
    output_manager.create_output_structure()

    # Check main directories
    for directory in output_manager.directories.values():
        assert directory.exists()
        assert directory.is_dir()

    # Check subdirectories
    assert (output_manager.directories["simulations"] / "mujoco").exists()
    assert (output_manager.directories["analysis"] / "biomechanics").exists()
    assert (output_manager.directories["reports"] / "pdf").exists()


def test_save_load_csv(output_manager):
    """Test saving and loading CSV results."""
    data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    filename = "test_data"

    # Save
    path = output_manager.save_simulation_results(
        data, filename, OutputFormat.CSV, engine="test"
    )

    assert path.exists()
    assert path.suffix == ".csv"

    # Load
    loaded = output_manager.load_simulation_results(
        path.name, OutputFormat.CSV, engine="test"
    )

    pd.testing.assert_frame_equal(data, loaded)


def test_save_load_json(output_manager):
    """Test saving and loading JSON results."""
    data = {"metadata": {"version": 1.0}, "results": [1.0, 2.0, 3.0]}
    filename = "test_json"

    # Save
    path = output_manager.save_simulation_results(
        data["results"],
        filename,
        OutputFormat.JSON,
        engine="test",
        metadata=data["metadata"],
    )

    assert path.exists()
    assert path.suffix == ".json"

    # Load
    loaded = output_manager.load_simulation_results(
        path.name, OutputFormat.JSON, engine="test"
    )

    # Load returns results part
    assert loaded == [1.0, 2.0, 3.0]


def test_get_simulation_list(output_manager):
    """Test retrieving simulation list."""
    # Create some dummy files
    output_manager.create_output_structure()
    mujoco_dir = output_manager.directories["simulations"] / "mujoco"
    (mujoco_dir / "sim1.csv").touch()
    (mujoco_dir / "sim2.json").touch()

    # Test specific engine
    files = output_manager.get_simulation_list("mujoco")
    assert len(files) == 2
    assert "sim1.csv" in files
    assert "sim2.json" in files

    # Test empty engine
    files = output_manager.get_simulation_list("drake")
    assert len(files) == 0


def test_export_analysis_report(output_manager):
    """Test exporting analysis report."""
    data = {"summary": "test", "value": 100}
    path = output_manager.export_analysis_report(data, "report1", "json")

    assert path.exists()
    assert path.parent.name == "json"
    assert "report1" in path.name


def test_cleanup_integration(output_manager):
    """Test cleanup with actual file mtime modification."""
    output_manager.create_output_structure()
    temp_dir = output_manager.directories["cache"] / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Create file
    # Create file and age it
    f = temp_dir / "test.tmp"
    f.touch()

    # Move import to top-level if possible, but local is fine for test isolation
    import os
    import time

    # Set time to 2 days ago
    two_days_ago = time.time() - (2 * 86400)
    os.utime(f, (two_days_ago, two_days_ago))

    count = output_manager.cleanup_old_files(max_age_days=30)
    assert count == 1
    assert not f.exists()
