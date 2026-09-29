"""Visible action outcomes and capture identity from the player's workflow (#9913)."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings
from src.tools.capture_rig.capture_library import CaptureLibrary
from src.tools.capture_rig.gui import CaptureRigWidget
from tests.tools.capture_rig.test_core import _bundle
from tests.tools.capture_rig.test_gui import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def widget_at(tmp_path: Path) -> CaptureRigWidget:
    _app()
    return CaptureRigWidget(
        settings=QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat)
    )


def test_missing_input_is_visible_without_opening_the_log(tmp_path: Path) -> None:
    widget = widget_at(tmp_path)
    widget.capture.session_edit.clear()
    widget.trigger("ingest")
    assert "Cannot" in widget.journey.message.text()
    assert "session" in widget.journey.message.text().lower()
    assert not widget.journey.progress.isVisible()
    widget.close()


def test_open_status_tracks_command_outcome_for_the_selected_capture(
    tmp_path: Path, monkeypatch
) -> None:
    from PyQt6.QtWidgets import QTreeWidget
    from src.tools.capture_rig.capture_activity import read_activity

    root = tmp_path / "take"
    root.mkdir()
    _bundle(root)
    widget = widget_at(tmp_path)
    widget.capture.set_session_path(root)
    widget.refresh_session()
    monkeypatch.setattr(widget, "command_for", lambda action: ["fake", action])
    monkeypatch.setattr(widget.runner, "run", lambda argv: None)
    widget.trigger("ingest")
    widget.journey.show_details()
    assert read_activity(root).actions[0].status == "running"
    widget._on_command_finished(1)
    dialog = widget.journey._details
    assert dialog is not None
    tables = dialog.findChildren(QTreeWidget)
    assert tables[1].topLevelItem(0).text(1) == "Failed"
    assert widget.process.estimator() in tables[1].topLevelItem(0).text(2)
    assert read_activity(root).actions[0].status == "failed"
    dialog.close()
    widget.close()


def test_help_search_and_link_navigate_to_existing_workflow(tmp_path: Path) -> None:
    from PyQt6.QtCore import QUrl
    from PyQt6.QtWidgets import QLineEdit, QTextBrowser

    widget = widget_at(tmp_path)
    widget.journey.show_help()
    dialog = widget.journey._help
    assert dialog is not None
    browser = dialog.findChild(QTextBrowser)
    search = dialog.findChild(QLineEdit)
    assert browser is not None and search is not None
    help_text = browser.toPlainText()
    assert "Example: Prepare a Swing for Review" in help_text
    assert "Example: Compare With an Instructor Reference" in help_text
    assert "Example: Prepare Calibrated Body-Model Analysis" in help_text
    assert "current body models do not fit a club segment" in help_text
    import re
    from src.tools.capture_rig.workflow import STEPS

    linked_steps = re.findall(r'href="step:([^"\s]+)"', browser.toHtml())
    assert linked_steps
    assert set(linked_steps) <= {step.key for step in STEPS}
    search.setText("detector")
    search.returnPressed.emit()
    assert browser.textCursor().selectedText().lower() == "detector"
    browser.anchorClicked.emit(QUrl("step:reconstruct"))
    assert "Reconstruct" in widget.journey.message.text()
    assert widget.panes.is_visible("inputs")
    dialog.close()
    widget.close()


def test_command_has_running_failed_and_retry_states(
    tmp_path: Path, monkeypatch
) -> None:
    widget = widget_at(tmp_path)
    monkeypatch.setattr(widget, "command_for", lambda action: ["fake", action])
    monkeypatch.setattr(widget.runner, "run", lambda argv: None)
    widget.trigger("ingest")
    assert "Running" in widget.journey.message.text()
    assert not widget.capture.isEnabled()
    widget._on_command_finished(1)
    assert "failed" in widget.journey.message.text().lower()
    assert widget.journey.retry_button.isEnabled()
    assert widget.capture.isEnabled()
    widget.close()


def test_capture_title_and_id_follow_library_selection_and_clear_on_error(
    tmp_path: Path,
) -> None:
    widget = widget_at(tmp_path)
    root = tmp_path / "swing-a"
    root.mkdir()
    _bundle(root)
    library = CaptureLibrary(tmp_path / "library")
    library.register(root)
    notes = library.update(root, title="Seven Iron — Practice")
    widget.capture.set_session_path(root)
    widget.refresh_session()
    assert "Seven Iron" in widget.journey.identity.text()
    assert notes.capture_id in widget.journey.identity.toolTip()
    widget.capture.set_session_path(tmp_path / "missing")
    widget.refresh_session()
    assert "Seven Iron" not in widget.journey.identity.text()
    assert widget.media is None
    assert widget.playback.current_view_name() is None
    assert widget.playback.current_set_name() is None
    assert widget.playback.variants.selected() == ()
    assert widget.playback._media is None
    assert not widget.playback.isEnabled()
    assert "No capture" in widget.playback.image.text()
    widget.close()
