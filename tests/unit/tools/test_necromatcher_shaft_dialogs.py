"""Reviewed shaft JSON is read and admitted off the Qt thread."""

import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import pytest
from tests.unit.motion_matching.test_shaft_observations import evidence

pytestmark = pytest.mark.unit


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    Qt = pytest.importorskip("PyQt6.QtWidgets")
    return Qt.QApplication.instance() or Qt.QApplication([])


def make_dialog(kind, session, tmp_path):
    if kind == "refit":
        from src.tools.necromatcher.refit_dialog import ResearchRefitDialog

        dialog = ResearchRefitDialog(
            "source",
            {
                "frame_indices": [0, 2],
                "coordinate_order": ["hip"],
                "coordinate_units": ["rad"],
                "recorded_options": None,
            },
            session,
        )
        dialog.identity.setText("new")
        dialog.scales.setText("1")
        return dialog
    from src.tools.necromatcher.video_dialog import VideoExportDialog

    return VideoExportDialog("source", session, library_root=tmp_path / "library")


def wait_for(app, condition):
    deadline = time.monotonic() + 3
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.002)
    assert condition()


@pytest.mark.parametrize("kind", ["refit", "video"])
def test_import_only_selects_path_then_read_parse_submit_on_background(
    app, tmp_path, monkeypatch, kind
):
    from PyQt6.QtWidgets import QFileDialog

    path = tmp_path / "reviewed.json"
    value = evidence()
    path.write_text(json.dumps(value.to_record()), encoding="utf-8")
    main = threading.get_ident()
    reads = []
    original = Path.read_text

    def read(self, *args, **kwargs):
        if self == path:
            reads.append(threading.get_ident())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    calls = []

    def submit(*args):
        assert threading.get_ident() != main
        calls.append(args)
        return {
            "run_id": "run",
            "source_fit_id": "source",
            "new_fit_id": "new",
            "status": "failed",
            "acceptance": "rejected",
            "message": "Test completion",
            "blockers": [],
            "control_available": True,
            "qualification": "monocular_research_hypothesis",
            "download_available": False,
            "execution_verified": False,
        }

    session = SimpleNamespace(
        submit=submit, view=lambda run: record, cancel=lambda run: record
    )
    record = {
        "run_id": "run",
        "source_fit_id": "source",
        "status": "failed",
        "acceptance": "rejected",
        "message": "Test completion",
        "blockers": [],
        "control_available": True,
        "qualification": "monocular_research_hypothesis",
        "download_available": False,
        "execution_verified": False,
    }
    dialog = make_dialog(kind, session, tmp_path)
    try:
        dialog.shaft_import.click()
        assert not reads and not calls
        assert dialog.shaft_remove.isEnabled()
        assert "uncalibrated" in dialog.shaft_status.text().lower()
        dialog.start.click()
        wait_for(app, lambda: dialog.run is not None)
        assert reads and all(thread != main for thread in reads)
        assert calls[0][-1] == value
        assert len(calls[0]) == (4 if kind == "refit" else 2)
        dialog.shaft_remove.click()
        assert dialog._shaft_path is None
        dialog.start.click()
        wait_for(app, lambda: len(calls) == 2)
        assert len(calls[-1]) == (3 if kind == "refit" else 1)
    finally:
        dialog.cleanup()
        dialog.close()


@pytest.mark.parametrize("kind", ["refit", "video"])
@pytest.mark.parametrize("problem", ["malformed", "missing", "foreign_source"])
def test_bad_or_foreign_selection_never_submits(
    app, tmp_path, monkeypatch, kind, problem
):
    from PyQt6.QtWidgets import QFileDialog

    path = tmp_path / "reviewed.json"
    if problem != "missing":
        path.write_text(
            json.dumps(evidence().to_record())
            if problem == "foreign_source"
            else '{"schema":"wrong"}',
            encoding="utf-8",
        )
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    session = SimpleNamespace(
        submit=lambda *args: pytest.fail("Bad evidence submitted")
    )
    dialog = make_dialog(kind, session, tmp_path)
    try:
        dialog.shaft_import.click()
        if problem == "foreign_source":
            dialog.source_fit_id = "other"
        dialog.start.click()
        wait_for(app, lambda: dialog._worker is None)
        assert dialog.run is None and dialog.status.text()
    finally:
        dialog.cleanup()
        dialog.close()
