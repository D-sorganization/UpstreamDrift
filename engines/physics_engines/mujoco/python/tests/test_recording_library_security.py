import importlib.util
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# List of modules to mock
MOCK_MODULES = [
    "mujoco",
    "numpy",
    "scipy",
    "matplotlib",
    "matplotlib.pyplot",
    "PyQt6",
    "PyQt6.QtWidgets",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "cv2",
    "imageio",
    "sklearn",
    "h5py",
    "PIL",
    "defusedxml",
    "defusedxml.ElementTree",
]


@pytest.fixture()
def isolated_library() -> Iterator[tuple]:
    """
    Import RecordingLibrary in isolation with mocked dependencies.
    Restores sys.modules afterwards to avoid side effects on other tests.
    """
    # Create mocks
    mocks = {name: MagicMock() for name in MOCK_MODULES}

    # Patch sys.modules. This context manager ensures changes are reverted.
    with patch.dict(sys.modules, mocks):
        # Locate source file
        repo_root = Path(__file__).resolve().parent.parent
        lib_path = repo_root / "mujoco_humanoid_golf/recording_library.py"

        if not lib_path.exists():
            lib_path = Path(
                "python/mujoco_humanoid_golf/recording_library.py"
            ).resolve()

        # Dynamic import
        module_name = "mujoco_humanoid_golf.recording_library"
        spec = importlib.util.spec_from_file_location(module_name, lib_path)
        if spec is None or spec.loader is None:
            msg = f"Cannot load {module_name} from {lib_path}"
            raise ImportError(msg)

        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

        yield mod.RecordingLibrary, mod.RecordingMetadata


@pytest.fixture()
def temp_library() -> Iterator[str]:
    """Create a temporary recording library."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_add_recording_path_traversal(isolated_library, temp_library) -> None:
    """Test that add_recording prevents path traversal."""
    RecordingLibrary, RecordingMetadata = isolated_library
    lib = RecordingLibrary(temp_library)

    # Create a source file
    source_path = Path(temp_library) / "source.json"
    source_path.write_text("{}")

    # Metadata with path traversal
    metadata = RecordingMetadata(
        filename="../pwned.txt", golfer_name="Hacker", club_type="Driver"
    )

    # Attempt to add recording
    # Should sanitize "../pwned.txt" to "pwned.txt" and succeed
    lib.add_recording(str(source_path), metadata, copy_to_library=True)

    # Check if file exists outside library (vulnerability check)
    pwned_path = Path(temp_library).parent / "pwned.txt"
    assert not pwned_path.exists(), (
        "Path traversal succeeded! File created outside library."
    )

    # Check if file exists inside library (sanitized)
    sanitized_path = Path(temp_library) / "pwned.txt"
    assert sanitized_path.exists(), "File was not saved with sanitized filename."


def test_add_recording_sanitization(isolated_library, temp_library) -> None:
    """Test that generated filenames are sanitized."""
    RecordingLibrary, RecordingMetadata = isolated_library
    lib = RecordingLibrary(temp_library)
    source_path = Path(temp_library) / "source.json"
    source_path.write_text("{}")

    metadata = RecordingMetadata(
        golfer_name="AC/DC",
        club_type="Iron",
        filename="",  # Force generation
    )

    lib.add_recording(str(source_path), metadata, copy_to_library=True)

    # Filename should not contain slash
    assert "/" not in metadata.filename
    assert "\\" not in metadata.filename
    # Check that it contains the sanitized version
    assert "AC_DC" in metadata.filename or "AC-DC" in metadata.filename


def test_add_recording_dot_dot_sanitization(isolated_library, temp_library) -> None:
    """Test that path traversal using '..' as filename raises ValueError."""
    RecordingLibrary, RecordingMetadata = isolated_library
    lib = RecordingLibrary(temp_library)
    source_path = Path(temp_library) / "source.json"
    source_path.write_text("{}")

    metadata = RecordingMetadata(
        filename="..", golfer_name="Hacker", club_type="Driver"
    )

    # This should raise ValueError because ".." is an invalid filename
    with pytest.raises(ValueError, match='Filename cannot be empty, ".", or ".."'):
        lib.add_recording(str(source_path), metadata, copy_to_library=True)


def test_delete_recording_security(isolated_library, temp_library) -> None:
    """Test that delete_recording restricts deletion to library files."""
    RecordingLibrary, RecordingMetadata = isolated_library
    lib = RecordingLibrary(temp_library)

    # Create external file
    external_dir = Path(temp_library).parent / "external"
    external_dir.mkdir(exist_ok=True)
    external_file = external_dir / "important.txt"
    external_file.write_text("DATA")

    # Add recording referencing external file (copy=False)
    metadata = RecordingMetadata(golfer_name="Hacker", club_type="Driver")
    rec_id = lib.add_recording(str(external_file), metadata, copy_to_library=False)

    # Verify DB has absolute path
    saved_meta = lib.get_recording(rec_id)
    assert Path(saved_meta.filename).resolve() == external_file.resolve()

    # Should return True (DB deletion success) but NOT delete the file (Security)
    result = lib.delete_recording(rec_id, delete_file=True)
    assert result is True

    # Check external file still exists
    assert external_file.exists(), "External file was deleted!"

    # Cleanup
    if external_file.exists():
        external_file.unlink()
    if external_dir.exists():
        external_dir.rmdir()


def test_valid_deletion(isolated_library, temp_library) -> None:
    """Test that valid files inside library are deleted."""
    RecordingLibrary, RecordingMetadata = isolated_library
    lib = RecordingLibrary(temp_library)
    source_path = Path(temp_library) / "source.json"
    source_path.write_text("{}")

    metadata = RecordingMetadata(golfer_name="User", club_type="Driver")
    rec_id = lib.add_recording(str(source_path), metadata, copy_to_library=True)

    # Verify file exists in library
    saved_meta = lib.get_recording(rec_id)
    lib_file = Path(temp_library) / saved_meta.filename
    assert lib_file.exists()

    # Delete
    result = lib.delete_recording(rec_id, delete_file=True)
    assert result is True

    # Verify file is gone
    assert not lib_file.exists()


def test_add_recording_complex_traversal(isolated_library, temp_library) -> None:
    """Test complex path traversal attempts."""
    RecordingLibrary, RecordingMetadata = isolated_library
    lib = RecordingLibrary(temp_library)
    source_path = Path(temp_library) / "source.json"
    source_path.write_text("{}")

    # Test nested traversal
    metadata = RecordingMetadata(
        filename="../../etc/passwd", golfer_name="Hacker", club_type="Driver"
    )

    # Should sanitize to just the filename part or raise error
    # Current implementation sanitizes using Path().name, so "passwd"
    lib.add_recording(str(source_path), metadata, copy_to_library=True)

    expected_path = Path(temp_library) / "passwd"
    assert expected_path.exists()
    assert metadata.filename == "passwd"

    # Test absolute path attempt
    abs_path = Path(temp_library).parent / "root_file.txt"
    metadata_abs = RecordingMetadata(
        filename=str(abs_path), golfer_name="Hacker", club_type="Driver"
    )

    # Should sanitize to "root_file.txt"
    lib.add_recording(str(source_path), metadata_abs, copy_to_library=True)

    expected_abs_sanitized = Path(temp_library) / "root_file.txt"
    assert expected_abs_sanitized.exists()
    assert metadata_abs.filename == "root_file.txt"


def test_save_file_to_library_success(isolated_library, temp_library) -> None:
    """Test that file is correctly saved to library."""
    RecordingLibrary, RecordingMetadata = isolated_library
    lib = RecordingLibrary(temp_library)

    content = '{"test": "data"}'
    source_path = Path(temp_library).parent / "source.json"
    source_path.write_text(content)

    metadata = RecordingMetadata(golfer_name="Pro", club_type="Driver")
    rec_id = lib.add_recording(str(source_path), metadata, copy_to_library=True)

    # Verify file existence and content
    saved_meta = lib.get_recording(rec_id)
    saved_path = Path(temp_library) / saved_meta.filename

    assert saved_path.exists()
    assert saved_path.read_text() == content


def test_delete_recording_external_file_skipped(isolated_library, temp_library) -> None:
    """Test that deleting a recording with external file does NOT delete the file."""
    RecordingLibrary, RecordingMetadata = isolated_library
    lib = RecordingLibrary(temp_library)

    # Create external file
    external_file = Path(temp_library).parent / "external_data.json"
    external_file.write_text("{}")

    # Add with copy_to_library=False
    metadata = RecordingMetadata(golfer_name="External", club_type="Driver")
    rec_id = lib.add_recording(str(external_file), metadata, copy_to_library=False)

    # Attempt delete
    lib.delete_recording(rec_id, delete_file=True)

    # Verify external file still exists
    assert external_file.exists(), "External file should strictly NOT be deleted"

    # Verify DB entry is gone
    assert lib.get_recording(rec_id) is None

    # Cleanup
    if external_file.exists():
        external_file.unlink()
