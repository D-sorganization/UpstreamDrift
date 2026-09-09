"""Header, status strip and workflow-grouped action grid (#9816)."""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("cv2")

from PyQt6.QtWidgets import QApplication, QPushButton

from src.tools.capture_rig import workflow
from src.tools.capture_rig.action_grid import ActionGrid, group_actions
from src.tools.capture_rig.gui import ALWAYS_ENABLED, CaptureRigWidget
from src.tools.capture_rig.header import (
    StatusStrip,
    TileStatus,
    cameras_text,
    recording_kind,
)
from src.tools.capture_rig.record_bar import Phase

pytestmark = [pytest.mark.unit, pytest.mark.ui]

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


# -- pure models ------------------------------------------------------------------
def test_group_actions_places_each_action_once_at_its_last_step() -> None:
    actions = tuple(a for a, _ in CaptureRigWidget._ACTIONS)
    groups = group_actions(workflow.STEPS, actions)
    placed = [a for _, group in groups for a in group]
    assert sorted(placed) == sorted(actions)
    by_title = dict(groups)
    assert "record" in by_title["Record or Import the Swing Take"]
    assert "record" not in by_title[workflow.INTRINSICS.title]
    assert set(by_title["Export to the Motion Pipeline"]) == {
        "export",
        "clip",
        "compare_takes",
        "multipicture",
    }
    assert set(by_title["Session"]) == set(ALWAYS_ENABLED)
    titles = [t for t, _ in groups]
    step_titles = [s.title for s in workflow.STEPS]
    assert titles[:-1] == [t for t in step_titles if t in titles]
    assert all(group for _, group in groups)


def test_group_actions_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        group_actions(workflow.STEPS, ("record", "record"))


def test_tile_status_texts() -> None:
    assert cameras_text(0) == "cameras: none bound"
    assert cameras_text(1) == "cameras: 1 bound"
    assert cameras_text(3) == "cameras: 3 bound"
    assert recording_kind(Phase.IDLE) == "neutral"
    assert recording_kind(Phase.COUNTDOWN) == "warning"
    assert recording_kind(Phase.RECORDING) == "record"
    status = TileStatus()
    assert status.last_take_text() == "last take: none yet"
    assert status.with_last_take(0).last_take_text() == "last take: written"
    failed = status.with_last_take(3)
    assert failed.last_take_text() == "last take: failed (exit 3)"
    assert failed.last_take_kind() == "warning"
    with pytest.raises(ValueError):
        TileStatus(cameras_bound=-1)


# -- widgets -----------------------------------------------------------------------
def test_status_strip_renders_chips() -> None:
    _app()
    strip = StatusStrip()
    strip.set_cameras(2)
    strip.set_recording(Phase.RECORDING, "● REC 00:01 / 00:10")
    strip.set_last_take(1)
    assert strip.cameras.text() == "cameras: 2 bound"
    assert strip.recording.text() == "● REC 00:01 / 00:10"
    assert strip.last_take.text() == "last take: failed (exit 1)"
    assert strip.recording.styleSheet() != strip.cameras.styleSheet()
    strip.set_recording(Phase.IDLE, "")
    assert strip.recording.text() == "recorder idle"


def test_action_grid_keeps_the_buttons_and_labels_the_steps() -> None:
    _app()
    buttons = {a: QPushButton(label) for a, label in CaptureRigWidget._ACTIONS}
    grid = ActionGrid(buttons)
    assert set(grid.section_labels) == {t for t, _ in grid.groups}
    assert buttons["record"].parent() is grid
    label = grid.section_labels["Record or Import the Swing Take"]
    assert label.text() == "Record or Import the Swing Take"
    assert "color" in grid.section_style()


def test_tile_header_tracks_cameras_recording_and_last_take() -> None:
    _app()
    widget = CaptureRigWidget()
    assert set(widget.buttons) == {a for a, _ in widget._ACTIONS}
    assert widget.header.session is widget.session_label
    assert widget.status_strip.recording.text() == "recorder idle"
    widget.record_bar.badge_changed.emit("Starting in 3")
    assert widget.status_strip.recording.text() == "Starting in 3"
    widget.preview.state_changed.emit(True)
    assert widget.status_strip.cameras.text() == "cameras: none bound"
    widget._take_running = True
    widget._on_command_finished(2)
    assert widget.status_strip.last_take.text() == "last take: failed (exit 2)"
    assert not widget._take_running
    widget.restyle()  # idempotent
    widget.shutdown()
