"""Native export controls follow service authority and preserve output bytes."""

from pathlib import Path
import threading

import pytest

pytestmark = pytest.mark.unit


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


def test_force_layer_is_off_by_default_and_forwarded_when_checked(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    from src.tools.necromatcher.video_dialog import VideoExportDialog

    app = QApplication.instance() or QApplication([])
    calls = []

    class Session:
        def submit(self, fit, **kwargs):
            calls.append(kwargs)
            raise ValueError("stop after recording the request")

    dialog = VideoExportDialog("fit", Session(), library_root=Path("library"))
    try:
        assert not dialog.force_layer.isChecked()
        assert dialog.force_layer_settings() is None
        dialog.force_layer.setChecked(True)
        dialog.segment_shading.setChecked(True)
        assert dialog.force_layer_settings() == {
            "enabled": True,
            "kinds": ["joint_reaction"],
            "scale": 1.0,
            "segment_shading": True,
        }
        dialog.start.click()
        dialog._worker.wait(2)
        dialog._poll()
        assert calls == [{"force_layer": dialog.force_layer_settings()}]
    finally:
        dialog.cleanup()
