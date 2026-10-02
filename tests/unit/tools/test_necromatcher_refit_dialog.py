"""Desktop research controls stay responsive during job submission."""

import threading
import time
import pytest

pytestmark = pytest.mark.unit


def test_refit_dialog_uses_shared_session_without_blocking_qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    from src.tools.necromatcher.refit_dialog import ResearchRefitDialog

    app = QApplication.instance() or QApplication([])
    entered, release = threading.Event(), threading.Event()
    ticks = []
    submissions = []
    record = {
        "run_id": "a" * 32,
        "source_fit_id": "source",
        "new_fit_id": "new",
        "status": "running",
        "acceptance": "partial",
        "blockers": [],
        "message": "Computing",
    }

    class Session:
        def submit(self, source, identity, options):
            submissions.append((source, identity, options))
            entered.set()
            if not release.wait(3):
                raise RuntimeError("Submission not released")
            return record

        def view(self, run):
            return record

        def cancel(self, run):
            record.update(status="cancelled", acceptance="interrupted")
            return record

    dialog = ResearchRefitDialog(
        "source",
        {
            "frame_indices": [0, 2],
            "coordinate_order": ["hip"],
            "coordinate_units": ["rad"],
            "recorded_options": None,
        },
        Session(),
    )
    dialog.identity.setText("new")
    dialog.scales.setText("1")
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    try:
        dialog.start.click()
        assert entered.wait(2)
        deadline = time.monotonic() + 0.1
        while time.monotonic() < deadline:
            app.processEvents()
        assert ticks
        release.set()
        deadline = time.monotonic() + 2
        while dialog.run is None and time.monotonic() < deadline:
            app.processEvents()
        assert dialog.run is not None
        assert submissions[0][:2] == ("source", "new")
        assert dialog.status.text().startswith("running · partial")
        dialog.cancel.click()
        assert dialog.status.text().startswith("cancelled · interrupted")
    finally:
        release.set()
        dialog.cleanup()
        timer.stop()
