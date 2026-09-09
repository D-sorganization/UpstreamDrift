"""The pre-ingestion editor is usable without a pose model or camera (#9860)."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import pytest

from src.motion_capture.rig.edits import CropRect, load_edits
from src.tools.capture_rig.swing_editor import SwingEditor
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_marks_crop_save_and_reopen(tmp_path: Path) -> None:
    _app()
    root = _bundle(tmp_path)
    dialog = SwingEditor(root)
    dialog.slider.setValue(2)
    dialog.mark_in.click()
    dialog.slider.setValue(4)
    dialog.mark_out.click()
    dialog.set_crop(CropRect(x=8, y=6, width=32, height=24))
    assert dialog.save()
    dialog.close()
    reopened = SwingEditor(root)
    assert (reopened.first.value(), reopened.last.value()) == (2, 4)
    assert reopened.current_edit().crop == CropRect(x=8, y=6, width=32, height=24)
    reopened.close()
    assert load_edits(root).views["a"].first == 2


def test_invalid_marks_stay_open_and_reset_restores_whole_view(tmp_path: Path) -> None:
    _app()
    root = _bundle(tmp_path)
    dialog = SwingEditor(root)
    dialog.first.setValue(4)
    dialog.last.setValue(1)
    assert not dialog.save()
    assert "Last frame" in dialog.status.text()
    dialog.reset_view()
    assert (dialog.first.value(), dialog.last.value()) == (0, 5)
    assert dialog.save()
    dialog.close()


def test_different_view_marks_survive_switching_and_fit_laptop(tmp_path: Path) -> None:
    app = _app()
    dialog = SwingEditor(_bundle(tmp_path))
    dialog.resize(850, 650)
    dialog.show()
    app.processEvents()
    dialog.first.setValue(2)
    dialog.view.setCurrentIndex(1)
    dialog.first.setValue(3)
    dialog.view.setCurrentIndex(0)
    assert dialog.first.value() == 2
    assert dialog.minimumSizeHint().width() <= 850
    assert dialog.canvas.width() >= 320
    assert dialog.save()
    dialog.close()


def test_drag_crop_uses_source_pixels_after_letterboxing(tmp_path: Path) -> None:
    app = _app()
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest

    dialog = SwingEditor(_bundle(tmp_path))
    dialog.resize(850, 650)
    dialog.show()
    app.processEvents()
    canvas = dialog.canvas
    scale = min(canvas.width() / 64, canvas.height() / 48)
    centre = QPoint(canvas.width() // 2, canvas.height() // 2)
    delta = QPoint(round(16 * scale), round(12 * scale))
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=centre - delta)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=centre + delta)
    crop = dialog.current_edit().crop
    assert crop is not None
    assert crop.x == pytest.approx(16, abs=1)
    assert crop.y == pytest.approx(12, abs=1)
    assert crop.width == pytest.approx(32, abs=1)
    assert crop.height == pytest.approx(24, abs=1)
    assert dialog.save()
    dialog.close()


def test_cancel_close_keeps_unsaved_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app()
    from PyQt6.QtWidgets import QMessageBox

    dialog = SwingEditor(_bundle(tmp_path))
    dialog.show()
    app.processEvents()
    dialog.first.setValue(2)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Cancel
    )
    dialog.close()
    assert dialog.isVisible() and dialog.first.value() == 2
    assert dialog.save()
    dialog.close()


def test_export_saves_selection_without_blocking_editor(tmp_path: Path) -> None:
    app = _app()
    editor = SwingEditor(_bundle(tmp_path))
    editor.first.setValue(1)
    editor.last.setValue(3)
    output = tmp_path / "selected.avi"
    editor.exporter.start(output)
    assert editor.exporter.busy
    deadline = monotonic() + 10
    while editor.exporter.busy and monotonic() < deadline:
        app.processEvents()
    assert not editor.exporter.busy
    assert output.is_file() and output.with_suffix(".json").is_file()
    assert "saved" in editor.status.text()
    editor.close()
