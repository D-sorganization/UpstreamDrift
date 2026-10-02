"""Native export controls follow service authority and preserve output bytes."""

from pathlib import Path
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit


@dataclass
class _Control:
    enabled: bool = False
    text: str = ""

    def setEnabled(self, value: bool) -> None:
        self.enabled = value

    def setText(self, value: str) -> None:
        self.text = value


def _stored_overlay_probe(status: str = "succeeded") -> SimpleNamespace:
    """Exercise control policy without constructing the Qt application."""
    return SimpleNamespace(
        run={
            "status": status,
            "acceptance": "rejected",
            "qualification": "monocular_research_hypothesis",
            "message": "Historical completion",
            "blockers": [],
            "control_available": False,
            "download_available": False,
            "execution_verified": True,
            "run_id": "owned",
            "producer_source_commit": "older-commit",
        },
        _closed=False,
        _worker=None,
        status=_Control(),
        start=_Control(),
        cancel=_Control(),
        save=_Control(),
        _timer=SimpleNamespace(stop=lambda: None),
    )


def test_historical_native_overlay_offers_explicit_guarded_verification() -> None:
    from src.tools.necromatcher.video_dialog import VideoExportDialog

    probe = _stored_overlay_probe()
    VideoExportDialog._render(probe)
    assert probe.save.enabled
    assert probe.save.text == "Verify Stored Overlay Package"
    assert "Readiness unverified" in probe.status.text
    assert "can reject changed files" in probe.status.text
    assert "older-commit" in probe.status.text


@pytest.mark.parametrize("status", ["failed", "cancelled", "running"])
def test_native_stored_verification_requires_successful_history(status: str) -> None:
    from src.tools.necromatcher.video_dialog import VideoExportDialog

    probe = _stored_overlay_probe(status)
    VideoExportDialog._render(probe)
    assert not probe.save.enabled


def test_native_unverified_package_uses_authoritative_guard_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tools.necromatcher.video_dialog import VideoExportDialog, QFileDialog

    probe = _stored_overlay_probe()
    operations: list[tuple[str, Any]] = []
    destination = tmp_path / "save.zip"
    probe.library_root = tmp_path / "library"
    probe._work = lambda operation, target: operations.append((operation, target))

    def reject_download(run: str) -> Path:
        assert run == "owned"
        raise ValueError("Stored artifact hash changed")

    probe.session = SimpleNamespace(download=reject_download)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args: (str(destination), "")
    )
    VideoExportDialog._save(probe)
    assert len(operations) == 1
    with pytest.raises(ValueError, match="hash changed"):
        operations[0][1]()
    assert not destination.exists()


def test_export_submission_is_responsive_and_download_requires_success(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from src.tools.necromatcher.video_dialog import VideoExportDialog

    app = QApplication.instance() or QApplication([])
    entered, release = threading.Event(), threading.Event()
    package = tmp_path / "checked.zip"
    package.write_bytes(b"checked-package")
    destination = tmp_path / "saved.zip"
    record = {
        "run_id": "run",
        "source_fit_id": "fit",
        "status": "running",
        "acceptance": "partial",
        "blockers": [],
        "message": "Rendering",
        "control_available": True,
        "download_available": False,
        "execution_verified": False,
        "qualification": "monocular_research_hypothesis",
    }

    class Session:
        def submit(self, fit):
            assert fit == "fit"
            entered.set()
            assert release.wait(3)
            return record.copy()

        def view(self, run):
            return record.copy()

        def cancel(self, run):
            record.update(status="cancelled", control_available=False)
            return record.copy()

        def download(self, run):
            assert record["status"] == "succeeded"
            return package

    dialog = VideoExportDialog("fit", Session(), library_root=tmp_path / "library")
    try:
        dialog.start.click()
        assert entered.wait(1)
        app.processEvents()
        assert not dialog.save.isEnabled()
        dialog.cancel.click()
        release.set()
        dialog._worker.wait(2)
        dialog._poll()
        assert dialog.run["status"] == "cancelled"
        assert not dialog.save.isEnabled()
        record.update(
            status="succeeded", download_available=True, control_available=False
        )
        dialog._cancel_requested = False
        dialog._poll()
        assert not dialog.save.isEnabled()
        record["execution_verified"] = True
        dialog._poll()
        assert dialog.save.isEnabled()
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *args: (str(destination), "")
        )
        dialog.save.click()
        dialog._worker.wait(2)
        dialog._poll()
        assert destination.read_bytes() == package.read_bytes()
        assert "research" in dialog.boundary.text().lower()
    finally:
        release.set()
        dialog.cleanup()


def test_copy_refuses_overwrite_and_library_destination(tmp_path):
    from src.tools.necromatcher.video_dialog import copy_export

    source = tmp_path / "source.zip"
    source.write_bytes(b"package")
    existing = tmp_path / "existing.zip"
    existing.write_bytes(b"owner")
    root = tmp_path / "library"
    root.mkdir()
    with pytest.raises(FileExistsError):
        copy_export(source, existing, root)
    with pytest.raises(ValueError, match="outside"):
        copy_export(source, root / "export.zip", root)
    assert existing.read_bytes() == b"owner"


def test_copy_rejects_corrupted_transfer_and_removes_only_its_output(
    tmp_path, monkeypatch
):
    from src.tools.necromatcher import video_dialog

    source = tmp_path / "checked.zip"
    destination = tmp_path / "copy.zip"
    source.write_bytes(b"verified-original-package")
    monkeypatch.setattr(
        video_dialog.shutil,
        "copyfileobj",
        lambda incoming, outgoing, size: outgoing.write(b"corrupted-transfer"),
    )
    with pytest.raises(ValueError, match="hash"):
        video_dialog.copy_export(source, destination, tmp_path / "library")
    assert not destination.exists()
    assert source.read_bytes() == b"verified-original-package"


def test_failed_export_cannot_save_and_cleanup_cancels_owned_job(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    from src.tools.necromatcher.video_dialog import VideoExportDialog

    app = QApplication.instance() or QApplication([])
    cancelled = []
    record = {
        "run_id": "owned",
        "source_fit_id": "fit",
        "status": "running",
        "acceptance": "partial",
        "qualification": "monocular_research_hypothesis",
        "message": "Working",
        "blockers": [],
        "control_available": True,
        "download_available": False,
    }

    class Session:
        def view(self, run_id):
            return record.copy()

        def cancel(self, run_id):
            cancelled.append(run_id)
            record.update(status="cancelled", control_available=False)
            return record.copy()

    dialog = VideoExportDialog("fit", Session(), library_root=tmp_path)
    dialog.run = record.copy()
    try:
        dialog._render()
        assert not dialog.save.isEnabled()
        record.update(status="failed", control_available=False, download_available=True)
        dialog._poll()
        assert not dialog.save.isEnabled()
        record.update(status="running", control_available=True)
        dialog.run = record.copy()
        dialog.cleanup()
        dialog.cleanup()
        app.processEvents()
        assert cancelled == ["owned"]
        assert not dialog._timer.isActive()
        previous = dialog.status.text()
        record.update(status="succeeded", download_available=True)
        dialog._poll()
        assert dialog.status.text() == previous
        assert not dialog.save.isEnabled()
    finally:
        dialog.cleanup()


def test_response_from_another_fit_or_run_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    from src.tools.necromatcher.video_dialog import VideoExportDialog

    app = QApplication.instance() or QApplication([])
    dialog = VideoExportDialog("fit", None, library_root=tmp_path)
    try:
        with pytest.raises(ValueError, match="selected fit"):
            dialog._check_owner(
                {"source_fit_id": "different", "run_id": "owned"}, "owned"
            )
        with pytest.raises(ValueError, match="selected fit"):
            dialog._check_owner(
                {"source_fit_id": "fit", "run_id": "different"}, "owned"
            )
        app.processEvents()
    finally:
        dialog.cleanup()
