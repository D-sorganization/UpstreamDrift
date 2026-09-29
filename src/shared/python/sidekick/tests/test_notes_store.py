"""Tests for notes_store.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from notes.models import NoteCard
from sidekick.notes_store import SidekickNotesStore


def test_init_validation(tmp_path: Path) -> None:
    """Test validation in constructor."""
    # Test project_root is None
    with pytest.raises(TypeError, match="project_root must be provided"):
        SidekickNotesStore(None)  # type: ignore[arg-type]

    # Test project_root does not exist
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="must exist and be a directory"):
        SidekickNotesStore(non_existent)

    # Test project_root is a file
    some_file = tmp_path / "some_file.txt"
    some_file.write_text("hello")
    with pytest.raises(ValueError, match="must exist and be a directory"):
        SidekickNotesStore(some_file)


def test_init_creates_dir(tmp_path: Path) -> None:
    """Test directory creation during init."""
    store = SidekickNotesStore(tmp_path)
    notes_dir = tmp_path / ".sidekick_notes"
    assert notes_dir.exists()
    assert notes_dir.is_dir()
    assert store._root == tmp_path


def test_create_note_validation(tmp_path: Path) -> None:
    """Test validation in create_note."""
    store = SidekickNotesStore(tmp_path)

    # Title is not str
    with pytest.raises(TypeError, match="title must be str"):
        store.create_note(123)  # type: ignore[arg-type]

    # Title is empty
    with pytest.raises(ValueError, match="title must not be empty"):
        store.create_note("")

    with pytest.raises(ValueError, match="title must not be empty"):
        store.create_note("   ")


def test_create_note_success(tmp_path: Path) -> None:
    """Test successful note creation."""
    store = SidekickNotesStore(tmp_path)
    note = store.create_note("Test Title", "Test Body", color="#ff0000")
    assert isinstance(note, NoteCard)
    assert note.title == "Test Title"
    assert note.markdown_body == "Test Body"
    assert note.color == "#ff0000"
    assert note.note_id != ""


def test_update_note_validation(tmp_path: Path) -> None:
    """Test validation in update_note."""
    store = SidekickNotesStore(tmp_path)

    # note_id validation
    with pytest.raises(TypeError, match="note_id must be str"):
        store.update_note(123, title="Title", body="Body")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="note_id must not be empty"):
        store.update_note("", title="Title", body="Body")

    # title validation
    with pytest.raises(TypeError, match="title must be str"):
        store.update_note("note_1", title=123, body="Body")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="title must not be empty"):
        store.update_note("note_1", title="", body="Body")


def test_update_note_success(tmp_path: Path) -> None:
    """Test successful note update."""
    store = SidekickNotesStore(tmp_path)
    note = store.create_note("Original Title", "Original Body")

    updated = store.update_note(
        note.note_id, title="New Title", body="New Body", color="#00ff00"
    )
    assert updated.note_id == note.note_id
    assert updated.title == "New Title"
    assert updated.markdown_body == "New Body"
    assert updated.color == "#00ff00"

    # Reload from store
    loaded = store.load_note(note.note_id)
    assert loaded is not None
    assert loaded.title == "New Title"


def test_list_notes(tmp_path: Path) -> None:
    """Test list_notes returns notes sorted properly or present in the list."""
    store = SidekickNotesStore(tmp_path)
    note1 = store.create_note("Note 1", "Body 1")
    note2 = store.create_note("Note 2", "Body 2")

    notes = store.list_notes()
    assert len(notes) == 2
    note_ids = {n.note_id for n in notes}
    assert note_ids == {note1.note_id, note2.note_id}


def test_load_note_validation(tmp_path: Path) -> None:
    """Test validation in load_note."""
    store = SidekickNotesStore(tmp_path)
    with pytest.raises(TypeError, match="note_id must be str"):
        store.load_note(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="note_id must not be empty"):
        store.load_note("")


def test_load_note_not_found(tmp_path: Path) -> None:
    """Test load_note when note does not exist."""
    store = SidekickNotesStore(tmp_path)
    assert store.load_note("non_existent_id") is None


def test_delete_note_validation(tmp_path: Path) -> None:
    """Test validation in delete_note."""
    store = SidekickNotesStore(tmp_path)
    with pytest.raises(TypeError, match="note_id must be str"):
        store.delete_note(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="note_id must not be empty"):
        store.delete_note("")


def test_delete_note_success(tmp_path: Path) -> None:
    """Test successful deletion/recycling of a note."""
    store = SidekickNotesStore(tmp_path)
    note = store.create_note("To Delete", "Body")
    assert store.load_note(note.note_id) is not None

    deleted = store.delete_note(note.note_id)
    assert deleted is True
    # The store deletes the file or moves it. Once deleted, it cannot be loaded.
    assert store.load_note(note.note_id) is None


def test_delete_note_not_found(tmp_path: Path) -> None:
    """Test deleting a note that does not exist."""
    store = SidekickNotesStore(tmp_path)
    assert store.delete_note("non_existent_id") is False
