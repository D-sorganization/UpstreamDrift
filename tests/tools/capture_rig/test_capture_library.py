"""Portable capture notes and reversible library operations (#9861)."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from src.motion_capture.rig.bundle import load_bundle
from src.motion_capture.rig.edits import SessionEdits, ViewEdit, save_edits
from src.tools.capture_rig.capture_library import CaptureLibrary, NOTES_FILE, read_notes
from tests.motion_capture.rig.test_ingest import _bundle

pytestmark = pytest.mark.unit


def test_notes_title_archive_and_search_survive_reopening(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    library = CaptureLibrary(tmp_path / "library")
    library.register(root)
    library.update(
        root,
        title="Driver lesson",
        notes="Keep the head steady.\nCheck finish.",
        archived=False,
    )
    reopened = CaptureLibrary(library.root)
    found = reopened.list(query="head steady")
    assert len(found) == 1 and found[0].title == "Driver lesson"
    assert found[0].notes.endswith("Check finish.")
    reopened.update(root, archived=True)
    assert reopened.list() == []
    assert reopened.list(archived=True)[0].root == root.resolve()
    reopened.update(root, archived=False)
    assert len(reopened.list()) == 1
    assert (root / "a_1.avi").is_file()


def test_rebuild_keeps_notes_and_duplicate_registration_is_idempotent(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    library = CaptureLibrary(tmp_path / "library")
    first = library.register(root)
    library.update(root, notes="tempo")
    assert library.register(root).capture_id == first.capture_id
    assert len(library.list()) == 1
    rebuilt = CaptureLibrary(tmp_path / "other-index")
    assert rebuilt.register(root).notes == "tempo"


def test_bad_entry_does_not_hide_other_captures(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    library = CaptureLibrary(tmp_path / "library")
    library.register(root)
    second = library.editable_copy(root)
    (root / NOTES_FILE).write_text("{broken", encoding="utf-8")
    rows = library.list()
    assert len(rows) == 2
    assert next(r for r in rows if r.root == root).problem
    assert not next(r for r in rows if r.root == second).problem


def test_editable_copy_preserves_sources_and_timing_but_not_analyses(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    library = CaptureLibrary(tmp_path / "library")
    library.register(root)
    library.update(root, notes="Original lesson")
    save_edits(root, SessionEdits(views={"a": ViewEdit(first=1, last=4)}))
    (root / "observations").mkdir()
    (root / "observations" / "a.json").write_text("{}", encoding="utf-8")
    original = (root / "recordings.json").read_bytes()
    copy = library.editable_copy(root)
    assert not (copy / "observations").exists()
    assert (root / "recordings.json").read_bytes() == original
    index = load_bundle(copy)[1]
    assert Path(index.recordings[0].file) == (root / "a_1.avi").resolve()
    assert read_notes(copy).notes == "Original lesson"
    assert read_notes(copy).source_capture == str(root.resolve())
    save_edits(copy, SessionEdits(views={"a": ViewEdit(first=2, last=4)}))
    row = next(r for r in library.list() if r.root == copy)
    assert row.external_bytes > 0


def test_rename_updates_index_and_refuses_collision_or_external_source(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    library = CaptureLibrary(tmp_path / "library")
    library.register(root)
    library.rename_recording(root, "a", "driver.avi")
    assert (root / "driver.avi").is_file() and not (root / "a_1.avi").exists()
    assert load_bundle(root)[1].recordings[0].file == "driver.avi"
    with pytest.raises(ValueError):
        library.rename_recording(root, "a", "b_2.avi")
    with pytest.raises(ValueError):
        library.rename_recording(root, "a", "../escape.avi")
    copy = library.editable_copy(root)
    with pytest.raises(ValueError, match="external"):
        library.rename_recording(copy, "a", "external.avi")


def test_rename_rolls_back_when_index_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path)
    library = CaptureLibrary(tmp_path / "library")
    library.register(root)
    from src.tools.capture_rig import capture_library

    real_write = capture_library.write_document

    def fail(path: Path, payload: dict) -> None:
        if path.name == "recordings.json":
            raise OSError("disk unavailable")
        real_write(path, payload)

    monkeypatch.setattr(capture_library, "write_document", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        library.rename_recording(root, "a", "renamed.avi")
    assert (root / "a_1.avi").exists() and not (root / "renamed.avi").exists()
    assert load_bundle(root)[1].recordings[0].file == "a_1.avi"


def test_interrupted_rename_recovers_without_overwriting_other_files(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    library = CaptureLibrary(tmp_path / "library")
    library.register(root)
    journal = root / ".recording-rename.json"
    journal.write_text(
        json.dumps(
            {
                "schema_version": "recording-rename/1.0.0",
                "view": "a",
                "source": "a_1.avi",
                "target": "renamed.avi",
            }
        ),
        encoding="utf-8",
    )
    (root / "a_1.avi").rename(root / "renamed.avi")
    library.recover_rename(root)
    assert (root / "a_1.avi").is_file() and not journal.exists()
    journal.write_text(
        json.dumps(
            {
                "schema_version": "recording-rename/1.0.0",
                "view": "a",
                "source": "../outside.avi",
                "target": "renamed.avi",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        library.recover_rename(root)
    assert (root / "a_1.avi").is_file()


def test_catalog_operations_release_database_handles(tmp_path: Path) -> None:
    library = CaptureLibrary(tmp_path / "library")
    library.register(_bundle(tmp_path))
    library.list()
    path = library.root / "capture_catalog.sqlite3"
    moved = path.with_suffix(".backup")
    path.rename(moved)  # Windows refuses this when a connection remains open.
    moved.rename(path)
