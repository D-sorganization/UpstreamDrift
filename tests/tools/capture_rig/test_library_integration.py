"""Capture header entry points respect the shared recording lifecycle (#9861)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.capture_rig.gui import CaptureRigWidget
from src.tools.capture_rig.record_bar import Phase
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_pane_layout import _app, _settings

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_header_library_and_editor_follow_session_and_recording_state(
    tmp_path: Path,
) -> None:
    _app()
    widget = CaptureRigWidget(settings=_settings(tmp_path))
    assert widget.library_actions.library_button.text() == "Library"
    assert widget.library_actions.library_button.isEnabled()
    assert not widget.library_actions.edit_button.isEnabled()
    root = _bundle(tmp_path)
    widget._open_library_capture(root)
    assert widget.library_actions.edit_button.isEnabled()
    widget._open_library_capture(tmp_path / "missing")
    assert not widget.library_actions.edit_button.isEnabled()
    widget.record_bar.clock.phase = Phase.COUNTDOWN
    widget.library_actions.refresh()
    assert not widget.library_actions.library_button.isEnabled()
    assert not widget.library_actions.edit_button.isEnabled()
    widget.record_bar.clock.phase = Phase.IDLE
    widget.shutdown()
