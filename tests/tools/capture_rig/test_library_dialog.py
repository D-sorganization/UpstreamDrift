"""Visible library controls keep notes and library navigation coherent (#9861)."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import pytest

from src.tools.capture_rig.capture_library import CaptureLibrary, read_notes
from src.tools.capture_rig.library_dialog import LibraryDialog
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def _ready(dialog: LibraryDialog) -> None:
    app = _app()
    end = monotonic() + 10
    while dialog.loading and monotonic() < end:
        app.processEvents()
    assert not dialog.loading
    app.processEvents()


def test_save_notes_open_and_archive_restore(tmp_path: Path) -> None:
    _app()
    root = _bundle(tmp_path)
    library = CaptureLibrary(tmp_path / "library")
    library.register(root)
    opened: list[Path] = []
    dialog = LibraryDialog(library, open_capture=opened.append)
    _ready(dialog)
    dialog.table.selectRow(0)
    dialog.title.setText("Lesson 1")
    dialog.notes.setPlainText("Watch head position\nFinish balanced")
    assert dialog.save_notes()
    assert read_notes(root).notes.endswith("Finish balanced")
    dialog.show()
    dialog.open_selected()
    assert opened == [root]
    assert not dialog.isVisible()
    dialog.archive_selected()
    _ready(dialog)
    assert dialog.table.rowCount() == 0
    dialog.archived.setChecked(True)
    _ready(dialog)
    dialog.table.selectRow(0)
    dialog.archive_selected()
    _ready(dialog)
    assert not read_notes(root).archived
    dialog.close()


def test_library_fits_laptop_and_shows_storage_distinction(tmp_path: Path) -> None:
    app = _app()
    library = CaptureLibrary(tmp_path / "library")
    library.register(_bundle(tmp_path))
    dialog = LibraryDialog(library, open_capture=lambda root: None)
    _ready(dialog)
    dialog.resize(900, 650)
    dialog.show()
    app.processEvents()
    assert dialog.minimumSizeHint().width() <= 900
    assert dialog.table.horizontalHeaderItem(1).text() == "Session storage"
    assert dialog.table.horizontalHeaderItem(2).text() == "Linked media"
    dialog.close()


def test_cancelled_navigation_keeps_unsaved_notes_and_archive_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    from PyQt6.QtWidgets import QMessageBox

    library = CaptureLibrary(tmp_path / "library")
    root = _bundle(tmp_path)
    library.register(root)
    library.editable_copy(root)
    dialog = LibraryDialog(library, open_capture=lambda path: None)
    _ready(dialog)
    dialog.table.selectRow(0)
    dialog.notes.setPlainText("Unsaved coaching note")
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Cancel
    )
    dialog.table.selectRow(1)
    assert dialog.table.currentRow() == 0
    assert dialog.notes.toPlainText() == "Unsaved coaching note"
    dialog.archived.setChecked(True)
    assert not dialog.archived.isChecked()
    assert dialog.save_notes()
    dialog.close()
