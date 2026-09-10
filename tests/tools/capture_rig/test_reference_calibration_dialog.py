"""Actual worker-backed setup, save/reopen and visible failure recovery."""

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from src.motion_capture.rig.capture_notes import CaptureNotes
from src.motion_capture.rig.documents import write_document
from src.motion_capture.rig.plan import CameraBinding, CaptureMode, RigPlan
from src.tools.capture_rig.reference_calibration.dialog import (
    ReferenceCalibrationDialog,
)

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def make_dialog(qtbot, root: Path) -> ReferenceCalibrationDialog:
    write_document(
        root / "capture_notes.json",
        CaptureNotes(capture_id="practice-1", title="Paper Practice").model_dump(
            mode="json"
        ),
    )
    plan = RigPlan(
        name="Reference",
        cameras=(
            CameraBinding(
                view="front",
                serial="front-serial",
                mode=CaptureMode(width=640, height=480),
            ),
        ),
    )
    dialog = ReferenceCalibrationDialog(plan, root, root / "profiles")
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.start.isEnabled(), timeout=10000)
    return dialog


def test_player_can_create_save_and_reopen_an_unknown_optics_session(
    qtbot, tmp_path
) -> None:
    dialog = make_dialog(qtbot, tmp_path)
    dialog.start.click()
    qtbot.waitUntil(lambda: dialog.save.isEnabled(), timeout=10000)
    session = dialog._session
    assert session["cameras"][0]["setup"]["zoom"] == "unknown"
    assert dialog.tabs.currentIndex() == 1
    assert "not been solved" in dialog.status.text()
    revision = session["revision_id"]
    dialog.save.click()
    qtbot.waitUntil(lambda: not dialog.client.busy, timeout=30000)
    assert "Saved" in dialog.status.text(), dialog.status.text()
    assert not dialog.save.isEnabled()
    dialog._request("load", revision_id=revision)
    qtbot.waitUntil(lambda: not dialog.client.busy, timeout=10000)
    assert "Opened" in dialog.status.text()
    assert dialog._session["revision_id"] == revision
    dialog.again.click()
    qtbot.waitUntil(lambda: dialog.save.isEnabled(), timeout=10000)
    assert dialog._session["revision_id"] != revision
    assert (tmp_path / "reference_calibration" / f"{revision}.json").is_file()
    dialog.placements.add_placement.click()
    assert dialog.placements.placement.currentText() == "Placement 1"
    dialog._dirty = False
    dialog.reject()


def test_missing_recording_provides_feedback_and_keeps_session(qtbot, tmp_path) -> None:
    dialog = make_dialog(qtbot, tmp_path)
    dialog.start.click()
    qtbot.waitUntil(lambda: dialog.save.isEnabled(), timeout=10000)
    before = dialog._session
    dialog._open_frame({"view": "front", "frame_index": 0})
    qtbot.waitUntil(lambda: not dialog.client.busy, timeout=30000)
    assert "try again" in dialog.status.text()
    assert dialog._session == before
    assert dialog.tabs.isEnabled()
    dialog._dirty = False
    dialog.reject()


def test_restore_clears_unrelated_profile_and_recovers_capture_profile(qtbot, tmp_path):
    from tests.tools.capture_rig.test_calibration_profiles import profile

    dialog = make_dialog(qtbot, tmp_path)
    dialog.start.click()
    qtbot.waitUntil(lambda: dialog.save.isEnabled(), timeout=10000)
    session = dialog._session
    panel = dialog.panels[0]
    saved = profile(tmp_path)
    panel.profiles.addItem(saved.name, saved)
    panel.profiles.setCurrentIndex(1)
    dialog._sync_setup(session)
    assert panel.profiles.currentData() is None
    panel.profiles.removeItem(1)
    captured = {
        **session,
        "cameras": [
            {**session["cameras"][0], "profile": saved.model_dump(mode="json")}
        ],
    }
    dialog._sync_setup(captured)
    assert panel.profiles.currentData().profile_id == saved.profile_id
    assert not panel.confirmed.isChecked()
    dialog._dirty = False
    dialog.reject()


def test_small_window_keeps_pages_scrollable_and_help_matches_the_guide(
    qtbot, tmp_path
):
    from PyQt6.QtWidgets import QScrollArea, QTextBrowser
    from src.tools.capture_rig.reference_calibration.guidance import GUIDE

    dialog = make_dialog(qtbot, tmp_path)
    dialog.resize(640, 560)
    qtbot.wait(20)
    assert dialog.width() <= 640 and dialog.height() <= 560
    for index in range(3):
        assert isinstance(dialog.tabs.widget(index), QScrollArea)
    help_view = dialog.tabs.widget(3)
    assert isinstance(help_view, QTextBrowser)
    assert "Optical zoom changes" in help_view.toPlainText()
    guide = (
        Path(__file__).resolve().parents[3]
        / "docs/motion_capture/common_reference_calibration.md"
    )
    assert guide.read_text(encoding="utf-8") == GUIDE.rstrip() + "\n"
    dialog.reject()


def test_original_frame_to_marked_saved_placement_through_real_worker(
    qtbot, tmp_path, monkeypatch
) -> None:
    from PyQt6.QtWidgets import QDialog
    from src.motion_capture.rig.bundle import load_bundle
    from src.motion_capture.rig.edits import (
        CropRect,
        SessionEdits,
        ViewEdit,
        save_edits,
    )
    from src.tools.capture_rig.reference_calibration.point_editor import (
        ReferencePointEditor,
    )
    from src.tools.capture_rig.reference_calibration.frame_selector import (
        ReferenceFrameSelector,
    )
    from tests.motion_capture.rig.test_ingest import _bundle

    root = _bundle(tmp_path)
    plan, _, _ = load_bundle(root)
    write_document(
        root / "capture_notes.json",
        CaptureNotes(capture_id="real-video", title="Practice Swing").model_dump(
            mode="json"
        ),
    )
    save_edits(
        root,
        SessionEdits(
            views={
                "a": ViewEdit(
                    first=1, last=3, crop=CropRect(x=8, y=6, width=31, height=23)
                )
            }
        ),
    )
    dialog = ReferenceCalibrationDialog(plan, root, tmp_path / "profiles")
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: dialog.start.isEnabled(), timeout=10000)
    dialog.start.click()
    qtbot.waitUntil(lambda: dialog.save.isEnabled(), timeout=10000)

    def mark(editor):
        assert editor._frame.shape[:2] == (48, 64)
        editor.mark(10, 10)
        editor.mark(30, 10)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ReferencePointEditor, "exec", mark)

    def choose(selector):
        qtbot.waitUntil(lambda: selector.use.isEnabled(), timeout=10000)
        selector.slider.setValue(2)
        qtbot.waitUntil(
            lambda: selector.displayed_index == 2 and selector.use.isEnabled(),
            timeout=10000,
        )
        assert not list((root / "reference_calibration" / "frames").glob("*.png"))
        selector.use.click()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ReferenceFrameSelector, "exec", choose)
    dialog._choose_frame({"view": "a", "frame_index": 0})
    qtbot.waitUntil(lambda: len(dialog._session["samples"]) == 1, timeout=30000)
    sample = dialog._session["samples"][0]
    assert sample["observation"]["point_ids"] == ["origin", "along-arrow"]
    assert sample["observation"]["frame_sequence"] == 2
    assert dialog.placements.frame_number.value() == 2
    assert (root / sample["source_frame"]).is_file()
    dialog._request(
        "target",
        parameters={
            "reference_id": "measured-My Paper",
            "shape": "rectangle",
            "width_m": 0.216,
            "length_m": 0.280,
        },
    )
    qtbot.waitUntil(lambda: not dialog.client.busy, timeout=10000)
    assert dialog._session["targets"][-1]["object_points_m"][2] == [0.28, 0.0, 0.216]
    dialog.save.click()
    qtbot.waitUntil(lambda: not dialog.client.busy, timeout=30000)
    assert "Saved" in dialog.status.text()
    dialog.panels[0].fields["zoom"].setText("New Zoom Position")
    assert not dialog.tabs.isTabEnabled(1)
    assert not dialog.tabs.isTabEnabled(2)
    assert "settings changed" in dialog.status.text()
    dialog.reject()
