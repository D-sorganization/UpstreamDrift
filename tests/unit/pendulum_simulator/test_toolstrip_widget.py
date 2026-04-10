"""Tests for the pendulum toolstrip widget."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from src.shared.python.pendulum_simulator.gui.toolstrip_widget import ToolStrip


def test_toolstrip_updates_playback_state_and_status(qtbot) -> None:
    widget = ToolStrip()
    qtbot.addWidget(widget)

    scrubbed: list[int] = []
    widget.frame_scrubbed.connect(scrubbed.append)

    widget.set_frame_range(5)
    widget.set_frame(2)
    assert widget._frame_lbl.text() == "50% (2/4)"

    widget._on_frame_slider_changed(3)
    assert scrubbed[-1] == 3
    assert widget._frame_lbl.text() == "75% (3/4)"

    widget.set_running(True)
    assert not widget.btn_run.isEnabled()
    assert not widget.btn_reset.isEnabled()
    assert widget._status_lbl.text() == "Simulating…"

    widget.set_running(False)
    assert widget.btn_run.isEnabled()
    assert widget.btn_reset.isEnabled()
    assert widget._status_lbl.text() == "Ready"

    widget.stop_play()
    assert widget.btn_play.text() == "▶ Play"
    assert not widget.btn_play.isChecked()


def test_toolstrip_rebuilds_segment_controls(qtbot) -> None:
    widget = ToolStrip()
    qtbot.addWidget(widget)

    emitted: list[object] = []
    widget.segment_visibility_changed.connect(emitted.append)

    widget.set_segment_names([("hip", "Hip"), ("wrist", "Wrist")])
    assert widget._segment_names == ["hip", "wrist"]
    assert list(widget._segment_checks) == ["hip", "wrist"]
    assert emitted[-1] is None

    widget._segment_checks["hip"].setChecked(False)
    assert emitted[-1] == {"wrist"}
