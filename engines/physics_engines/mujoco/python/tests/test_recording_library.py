import sqlite3
import sys
import types
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def pkg_mocks() -> dict[str, MagicMock]:
    """Provide mocks for heavy dependencies."""
    return {
        "mujoco": MagicMock(),
        "scipy": MagicMock(),
        "scipy.spatial": MagicMock(),
        "scipy.optimize": MagicMock(),
        "scipy.interpolate": MagicMock(),
        "scipy.signal": MagicMock(),
        "scipy.linalg": MagicMock(),
        "scipy.spatial.transform": MagicMock(),
        "matplotlib": MagicMock(),
        "matplotlib.pyplot": MagicMock(),
        "matplotlib.animation": MagicMock(),
        "matplotlib.figure": MagicMock(),
        "matplotlib.backends": MagicMock(),
        "matplotlib.backends.backend_qtagg": MagicMock(),
        "pinocchio": MagicMock(),
        "PyQt6": MagicMock(),
        "PyQt6.QtWidgets": MagicMock(),
        "PyQt6.QtCore": MagicMock(),
        "PyQt6.QtGui": MagicMock(),
    }


@pytest.fixture(scope="module")
def library_module(
    pkg_mocks: dict[str, MagicMock],
) -> Generator[types.ModuleType, None, None]:
    """Import the library module with mocks in place."""
    with patch.dict(sys.modules, pkg_mocks):
        # Import inside the patch
        # This prevents global pollution of sys.modules for other tests
        import mujoco_golf_pendulum.recording_library as mod

        yield mod


@pytest.fixture()
def recording_lib(
    library_module: types.ModuleType, tmp_path: Path
) -> Any:  # noqa: ANN401
    """Create a temporary recording library."""
    lib_dir = tmp_path / "recordings"
    return library_module.RecordingLibrary(str(lib_dir))


def test_add_recording_sanitization(
    library_module: types.ModuleType, recording_lib: Any, tmp_path: Path  # noqa: ANN401
) -> None:
    """Test that filenames are sanitized when adding recordings."""
    RecordingMetadata = library_module.RecordingMetadata

    # Create dummy data file
    data_file = tmp_path / "test_data.json"
    data_file.write_text("{}")

    # Try to add with path traversal filename
    metadata = RecordingMetadata(golfer_name="Test", filename="../hacked.json")

    # This should succeed but sanitize the filename
    recording_lib.add_recording(str(data_file), metadata)

    # Verify the file was created in the library, not outside
    assert (recording_lib.library_path / "hacked.json").exists()
    assert not (recording_lib.library_path.parent / "hacked.json").exists()

    # Verify metadata stored in DB
    rec = recording_lib.search_recordings(golfer_name="Test")[0]
    assert rec.filename == "hacked.json"

    # Verify validation for invalid filenames
    bad_metadata = RecordingMetadata(golfer_name="Bad", filename="..")
    with pytest.raises(ValueError, match="Filename cannot be"):
        recording_lib.add_recording(str(data_file), bad_metadata)


def test_delete_recording_security(recording_lib: Any) -> None:  # noqa: ANN401
    """Test that deleting a recording outside the library is prevented."""
    # Manually inject a malicious record into DB
    conn = sqlite3.connect(str(recording_lib.db_path))
    cursor = conn.cursor()
    cursor.execute("INSERT INTO recordings (filename) VALUES (?)", ("../target.txt",))
    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Create target file outside library
    target_file = recording_lib.library_path.parent / "target.txt"
    target_file.write_text("data")

    # Try to delete
    recording_lib.delete_recording(rec_id, delete_file=True)

    # File should still exist
    assert target_file.exists()


def test_normal_operations(
    library_module: types.ModuleType, recording_lib: Any, tmp_path: Path  # noqa: ANN401
) -> None:
    """Test normal add/get/delete operations."""
    RecordingMetadata = library_module.RecordingMetadata

    data_file = tmp_path / "normal.json"
    data_file.write_text("{}")

    metadata = RecordingMetadata(golfer_name="Pro", club_type="Driver")
    rec_id = recording_lib.add_recording(str(data_file), metadata)

    rec = recording_lib.get_recording(rec_id)
    assert rec.golfer_name == "Pro"

    # Verify file was copied
    lib_file = recording_lib.library_path / rec.filename
    assert lib_file.exists()
    assert lib_file.name.startswith("20")  # Check timestamp prefix if generated

    assert recording_lib.delete_recording(rec_id, delete_file=True)
    assert recording_lib.get_recording(rec_id) is None
