"""Tests for src.shared.python.notes.models and notes.storage (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.shared.python.notes.models import RecycledNoteItem
from src.shared.python.notes.storage import NotesStorage

# ── RecycledNoteItem ─────────────────────────────────────────────────────────


class TestRecycledNoteItem:
    def test_notes_storage_construction(self) -> None:
        item = RecycledNoteItem(
            item_id="id1",
            reason="test",
            path="/tmp/a.txt",
            original_path="/proj/notes.txt",
            deleted_at="20250101T000000Z",
        )
        assert item.item_id == "id1"
        assert item.reason == "test"

    def test_frozen_raises_on_mutation(self) -> None:
        item = RecycledNoteItem("x", "r", "/a", "/b", "ts")
        with pytest.raises((AttributeError, TypeError)):
            item.item_id = "new"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = RecycledNoteItem("id", "r", "/a", "/b", "ts")
        b = RecycledNoteItem("id", "r", "/a", "/b", "ts")
        assert a == b

    def test_inequality(self) -> None:
        a = RecycledNoteItem("id1", "r", "/a", "/b", "ts")
        b = RecycledNoteItem("id2", "r", "/a", "/b", "ts")
        assert a != b


# ── NotesStorage ─────────────────────────────────────────────────────────────


class TestNotesStorageConstruction:
    def test_notes_storage_valid_construction(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        assert storage.project_dir == tmp_path

    def test_nonexistent_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="project_dir must exist"):
            NotesStorage("/nonexistent/path/does/not/exist")

    def test_empty_notes_filename_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="notes_filename cannot be empty"):
            NotesStorage(tmp_path, notes_filename="   ")

    def test_custom_notes_filename(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path, notes_filename="custom.txt")
        assert storage.notes_path.name == "custom.txt"

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        storage = NotesStorage(str(tmp_path))
        assert storage.project_dir == tmp_path


class TestNotesStorageLoadSave:
    def test_load_empty_when_file_absent(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        result = storage.load_text()
        assert result == ""

    def test_save_and_load(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        storage.save_text("Hello notes!")
        assert storage.load_text() == "Hello notes!"

    def test_save_returns_path(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        p = storage.save_text("data")
        assert isinstance(p, Path)
        assert p.exists()

    def test_save_none_raises(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        with pytest.raises(ValueError):
            storage.save_text(None)  # type: ignore[arg-type]

    def test_clear_empties_notes(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        storage.save_text("content")
        storage.clear()
        assert storage.load_text() == ""


class TestNotesStorageRecycleBin:
    def test_move_to_recycle_no_file_raises(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        with pytest.raises(FileNotFoundError):
            storage.move_to_recycle()

    def test_move_to_recycle_returns_item(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        storage.save_text("test content")
        item = storage.move_to_recycle(reason="test_delete")
        assert isinstance(item, RecycledNoteItem)
        assert item.reason == "test_delete"

    def test_recycled_item_has_valid_id(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        storage.save_text("data")
        item = storage.move_to_recycle()
        assert item.item_id != ""

    def test_list_recycled_empty_initially(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        assert storage.list_recycled() == []

    def test_list_recycled_after_move(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        storage.save_text("data")
        storage.move_to_recycle()
        items = storage.list_recycled()
        assert len(items) == 1

    def test_latest_recycled_id_none_initially(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        assert storage.latest_recycled_id() is None

    def test_latest_recycled_id_after_move(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        storage.save_text("data")
        item = storage.move_to_recycle()
        assert storage.latest_recycled_id() == item.item_id

    def test_restore_returns_path(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        storage.save_text("restore me")
        item = storage.move_to_recycle()
        restored = storage.restore(item.item_id)
        assert restored is not None
        assert restored.exists()

    def test_restore_nonexistent_returns_none(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        result = storage.restore("nonexistent_id")
        assert result is None

    def test_purge_removes_item(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        storage.save_text("purge me")
        item = storage.move_to_recycle()
        assert storage.purge(item.item_id) is True
        assert storage.list_recycled() == []

    def test_purge_nonexistent_returns_false(self, tmp_path: Path) -> None:
        storage = NotesStorage(tmp_path)
        assert storage.purge("nonexistent_id") is False
