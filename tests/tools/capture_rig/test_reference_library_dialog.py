"""Native reference import and management preserve user notes and source media."""

from pathlib import Path
from time import monotonic

import numpy as np
import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QMessageBox

from src.motion_capture.reference.importers import MotionDraft
from src.motion_capture.reference.storage import ReferenceLibrary
from src.tools.capture_rig.reference_import import (
    ReferenceMappingDialog,
    load_reference_video,
)
from src.tools.capture_rig.reference_library_dialog import ReferenceLibraryDialog
from tests.motion_capture.test_reference_assets import motion
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def settle(dialog: ReferenceLibraryDialog) -> None:
    deadline = monotonic() + 15
    while dialog._worker is not None and monotonic() < deadline:
        _app().processEvents()
        QTest.qWait(10)
    assert dialog._worker is None


def test_library_opens_model_analysis_without_a_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tools.capture_rig.model_analysis_dialog import ModelAnalysisDialog

    _app()
    library = ReferenceLibrary(tmp_path)
    asset = motion()
    library.save(asset)
    opened = []

    def inspect(dialog: ModelAnalysisDialog) -> int:
        opened.append(dialog.model_source.reader.recipe.asset.id)
        dialog.close()
        return 0

    monkeypatch.setattr(ModelAnalysisDialog, "exec", inspect)
    dialog = ReferenceLibraryDialog(library)
    settle(dialog)
    dialog.analyze_model()
    assert opened == [asset.id]
    assert not list(tmp_path.rglob("recordings.json"))
    dialog.close()


def test_notes_archive_and_reopen(tmp_path: Path) -> None:
    app = _app()
    library = ReferenceLibrary(tmp_path)
    asset = motion()
    library.save(asset)
    dialog = ReferenceLibraryDialog(library)
    dialog.show()
    settle(dialog)
    assert dialog.title.text() == asset.title
    dialog.title.setText("Expert lesson")
    dialog.notes.setPlainText("Keep the lead wrist flat")
    dialog.save_notes()
    settle(dialog)
    assert library.load(asset.id).notes == "Keep the lead wrist flat"
    dialog.archive_selected()
    settle(dialog)
    assert dialog.items.count() == 0
    dialog.archived.setChecked(True)
    settle(dialog)
    assert dialog.items.count() == 1
    assert dialog.title.text() == "Expert lesson"
    assert dialog.minimumSizeHint().width() <= 800
    dialog.close()
    app.processEvents()


def test_mapping_requires_explicit_confirmation() -> None:
    app = _app()
    asset = motion()
    draft = MotionDraft(
        asset.source, asset.source_names, (0, 1), np.zeros((2, 2, 3)), "mm", True
    )
    dialog = ReferenceMappingDialog(draft)
    dialog.show()
    dialog.accept()
    assert dialog.isVisible()
    assert "confirm" in dialog.status.text()
    dialog.units.setCurrentText("mm")
    for axis, box in zip(("+X", "+Y", "+Z"), dialog.axes, strict=True):
        box.setCurrentText(axis)
    dialog.confirm.setChecked(True)
    dialog.edges.setPlainText("hip, hand")
    assert dialog.options()["edges"] == ((0, 1),)
    dialog.accept()
    assert not dialog.isVisible()
    app.processEvents()


def test_reference_video_stays_linked_and_two_dimensional(tmp_path: Path) -> None:
    source = _bundle(tmp_path) / "a_1.avi"
    original = source.read_bytes()
    asset = load_reference_video(source)
    assert asset.kind == "video" and asset.source.path == str(source.resolve())
    assert asset.frames > 0 and asset.fps > 0
    assert source.read_bytes() == original


def test_unsaved_close_can_be_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app()
    library = ReferenceLibrary(tmp_path)
    library.save(motion())
    dialog = ReferenceLibraryDialog(library)
    dialog.show()
    settle(dialog)
    dialog.notes.setPlainText("Unsaved cue")
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Cancel
    )
    dialog.archived.setChecked(True)
    assert not dialog.archived.isChecked()
    assert dialog.notes.toPlainText() == "Unsaved cue"
    dialog.reject()
    app.processEvents()
    assert dialog.isVisible()
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Discard
    )
    dialog.close()
