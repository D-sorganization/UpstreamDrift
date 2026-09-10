"""Reuse review keeps confirmations, cancellation and next steps visible."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QPushButton
from src.motion_capture.rig.capture_notes import CaptureNotes
from src.motion_capture.rig.documents import write_document
from src.motion_capture.rig.plan import CameraBinding, RigPlan
from src.tools.capture_rig.calibration_dialog import CalibrationDialog
from src.tools.capture_rig.reference_calibration.reuse_dialog import (
    ReuseCalibrationDialog,
)

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def make_dialog(qtbot, tmp_path):
    write_document(
        tmp_path / "capture_notes.json",
        CaptureNotes(title="Driver Swing 2").model_dump(mode="json"),
    )
    plan = RigPlan(
        name="Test", cameras=(CameraBinding(view="front", serial="camera-1"),)
    )
    dialog = ReuseCalibrationDialog(plan, tmp_path, tmp_path)
    qtbot.addWidget(dialog)
    dialog.show()
    return dialog


def preview():
    return {
        "source_sha256": "a" * 64,
        "source_title": "Calibration Take",
        "source_capture_id": "source-1",
        "result": {
            "reviewed_utc": "2026-09-10T00:00:00+00:00",
            "scene_id": "Room 1",
            "anchor": {"translation_m": [0, 0, 0]},
            "profile_selections": [],
            "residuals": [],
            "limitations": ["Synthetic test only"],
        },
    }


def test_reuse_requires_both_checkboxes_and_reports_failure(qtbot, tmp_path):
    dialog = make_dialog(qtbot, tmp_path)
    assert not dialog.use.isEnabled()
    dialog._action = "inspect_reuse"
    dialog._completed(preview())
    assert "Calibration Take" in dialog.evidence.toPlainText()
    assert "review" in dialog.status.text().lower()
    dialog.settings.setChecked(True)
    assert not dialog.use.isEnabled()
    dialog.scene.setChecked(True)
    assert dialog.use.isEnabled()
    dialog._failed("Camera layout changed since review")
    assert "not applied" in dialog.status.text()
    assert not dialog.use.isEnabled()
    dialog.reject()


def test_cancel_cannot_apply_a_late_worker_result(qtbot, tmp_path, monkeypatch):
    dialog = make_dialog(qtbot, tmp_path)
    cancelled = []
    monkeypatch.setattr(dialog.client, "cancel", lambda: cancelled.append(True))
    dialog.reject()
    dialog._action = "adopt_layout"
    dialog._completed({"result_path": "late.json"})
    assert cancelled == [True]
    assert dialog.output_path is None


def test_small_reuse_review_and_entry_point_fit_standard_dialog(qtbot, tmp_path):
    dialog = make_dialog(qtbot, tmp_path)
    dialog.resize(640, 560)
    qtbot.wait(20)
    assert dialog.width() <= 640 and dialog.height() <= 560
    dialog.reject()
    plan = RigPlan(
        name="Test", cameras=(CameraBinding(view="front", serial="camera-1"),)
    )
    entry = CalibrationDialog(plan, tmp_path, reference_available=True)
    qtbot.addWidget(entry)
    button = next(
        item
        for item in entry.findChildren(QPushButton)
        if item.text() == "Reuse a Camera Layout…"
    )
    button.click()
    assert entry.reuse_requested and entry.output_path is None
