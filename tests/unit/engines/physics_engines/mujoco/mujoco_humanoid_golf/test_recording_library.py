"""Unit tests for recording_library.py."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.recording_library import (
    ConnectionPool,
    RecordingLibrary,
    RecordingMetadata,
    create_metadata_from_recording,
)


@pytest.fixture
def temp_lib_dir(tmp_path):
    lib_dir = tmp_path / "recordings"
    return lib_dir


@pytest.fixture
def recording_library(temp_lib_dir):
    library = RecordingLibrary(str(temp_lib_dir))
    yield library
    library.close()


@pytest.fixture
def dummy_data_file(temp_lib_dir):
    data_path = temp_lib_dir / "dummy_data.json"
    with open(data_path, "w") as f:
        json.dump({"times": [0.0, 0.1, 0.2], "club_head_speed": [0.0, 10.0, 20.0]}, f)
    return str(data_path)


def test_connection_pool(temp_lib_dir):
    """Test ConnectionPool functionality."""
    temp_lib_dir.mkdir(exist_ok=True)
    db_path = temp_lib_dir / "test.db"
    pool = ConnectionPool(str(db_path))
    conn = pool.get_connection()
    assert conn is not None
    pool.close_all()


def test_recording_metadata_defaults():
    """Test RecordingMetadata defaults."""
    metadata = RecordingMetadata()
    assert metadata.golfer_name == "Unknown"
    assert metadata.club_type == "Driver"
    assert metadata.swing_type == "Practice"
    assert metadata.rating == 0


def test_add_recording(recording_library, dummy_data_file):
    """Test adding a recording to the library."""
    metadata = RecordingMetadata(
        golfer_name="John Doe",
        club_type="Iron",
        swing_type="Practice",
    )
    rec_id = recording_library.add_recording(
        dummy_data_file, metadata, copy_to_library=True
    )

    assert rec_id > 0

    fetched = recording_library.get_recording(rec_id)
    assert fetched is not None
    assert fetched.golfer_name == "John Doe"
    assert fetched.club_type == "Iron"
    assert fetched.checksum != ""
    assert (Path(recording_library.library_path) / fetched.filename).exists()


def test_add_recording_no_copy(recording_library, dummy_data_file):
    """Test adding a recording without copying."""
    metadata = RecordingMetadata(golfer_name="Jane Doe")
    rec_id = recording_library.add_recording(
        dummy_data_file, metadata, copy_to_library=False
    )

    fetched = recording_library.get_recording(rec_id)
    assert fetched is not None
    assert fetched.filename == dummy_data_file


def test_update_recording(recording_library, dummy_data_file):
    """Test updating recording metadata."""
    metadata = RecordingMetadata(golfer_name="Test Update")
    rec_id = recording_library.add_recording(
        dummy_data_file, metadata, copy_to_library=False
    )

    fetched = recording_library.get_recording(rec_id)
    fetched.rating = 5
    fetched.notes = "Great swing"

    success = recording_library.update_recording(fetched)
    assert success is True

    updated = recording_library.get_recording(rec_id)
    assert updated.rating == 5
    assert updated.notes == "Great swing"


def test_delete_recording(recording_library, dummy_data_file):
    """Test deleting a recording."""
    metadata = RecordingMetadata(golfer_name="Test Delete")
    rec_id = recording_library.add_recording(
        dummy_data_file, metadata, copy_to_library=True
    )

    fetched = recording_library.get_recording(rec_id)
    file_path = recording_library.library_path / fetched.filename
    assert file_path.exists()

    success = recording_library.delete_recording(rec_id, delete_file=True)
    assert success is True

    assert recording_library.get_recording(rec_id) is None
    assert not file_path.exists()


def test_search_recordings(recording_library, dummy_data_file):
    """Test searching recordings."""
    metadata_1 = RecordingMetadata(
        golfer_name="Alice", club_type="Driver", rating=5, filename="alice.json"
    )
    metadata_2 = RecordingMetadata(
        golfer_name="Bob", club_type="Iron", rating=3, filename="bob.json"
    )

    recording_library.add_recording(dummy_data_file, metadata_1, copy_to_library=True)
    recording_library.add_recording(dummy_data_file, metadata_2, copy_to_library=True)

    results = recording_library.search_recordings(golfer_name="Alice")
    assert len(results) == 1
    assert results[0].golfer_name == "Alice"

    results = recording_library.search_recordings(club_type="Iron")
    assert len(results) == 1
    assert results[0].golfer_name == "Bob"

    results = recording_library.search_recordings(min_rating=4)
    assert len(results) == 1
    assert results[0].golfer_name == "Alice"


def test_get_statistics(recording_library, dummy_data_file):
    """Test getting statistics."""
    metadata_1 = RecordingMetadata(
        club_type="Driver", peak_club_speed=10.0, rating=5, filename="stat1.json"
    )
    metadata_2 = RecordingMetadata(
        club_type="Iron", peak_club_speed=20.0, rating=3, filename="stat2.json"
    )

    recording_library.add_recording(dummy_data_file, metadata_1, copy_to_library=True)
    recording_library.add_recording(dummy_data_file, metadata_2, copy_to_library=True)

    stats = recording_library.get_statistics()
    assert stats["total_recordings"] == 2
    assert stats["average_rating"] == 4.0
    assert stats["speed_stats"]["min"] == 10.0
    assert stats["speed_stats"]["max"] == 20.0
    assert stats["speed_stats"]["average"] == 15.0
    assert stats["by_club_type"]["Driver"] == 1
    assert stats["by_club_type"]["Iron"] == 1


def test_export_import_library(recording_library, dummy_data_file, tmp_path):
    """Test exporting and importing library."""
    metadata_1 = RecordingMetadata(golfer_name="ExportTest")
    recording_library.add_recording(dummy_data_file, metadata_1, copy_to_library=False)

    export_path = tmp_path / "export.json"
    recording_library.export_library(str(export_path))
    assert export_path.exists()

    # Create new library and import
    new_lib_dir = tmp_path / "new_recordings"
    new_lib = RecordingLibrary(str(new_lib_dir))

    new_lib.import_library(str(export_path), merge=False)
    recordings = new_lib.get_all_recordings()
    assert len(recordings) == 1
    assert recordings[0].golfer_name == "ExportTest"

    new_lib.close()


def test_get_unique_values(recording_library, dummy_data_file):
    """Test getting unique values for fields."""
    metadata_1 = RecordingMetadata(
        golfer_name="Alice", club_type="Driver", filename="unique1.json"
    )
    metadata_2 = RecordingMetadata(
        golfer_name="Bob", club_type="Driver", filename="unique2.json"
    )

    recording_library.add_recording(dummy_data_file, metadata_1, copy_to_library=True)
    recording_library.add_recording(dummy_data_file, metadata_2, copy_to_library=True)

    golfers = recording_library.get_unique_values("golfer_name")
    assert sorted(golfers) == ["Alice", "Bob"]

    clubs = recording_library.get_unique_values("club_type")
    assert clubs == ["Driver"]


def test_create_metadata_from_recording():
    """Test create_metadata_from_recording function."""
    data_dict = {
        "times": [0.0, 1.0, 2.0],
        "club_head_speed": [0.0, 15.0, 30.0],
        "model_name": "TestModel",
    }

    metadata = create_metadata_from_recording(
        data_dict, golfer_name="Charlie", club_type="Putter"
    )

    assert metadata.golfer_name == "Charlie"
    assert metadata.club_type == "Putter"
    assert metadata.duration == 2.0
    assert metadata.peak_club_speed == 30.0
    assert metadata.num_frames == 3
    assert metadata.model_name == "TestModel"
