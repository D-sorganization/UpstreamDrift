"""Responsive native export controls over the shared isolated video service."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import os
import shutil
from typing import Any

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from src.shared.python.ui.adapters import BackgroundWorker, get_worker_adapter
from src.shared.python.workspace import compute_file_sha256


def copy_export(source: Path, destination: Path, library_root: Path) -> Path:
    """Copy checked output exclusively outside the immutable workspace."""
    destination = destination.resolve()
    if destination.is_relative_to(library_root.resolve()):
        raise ValueError("Save exports outside the library")
    expected_hash = compute_file_sha256(source)
    created = False
    try:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            created = True
            shutil.copyfileobj(incoming, outgoing, 65536)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if compute_file_sha256(destination) != expected_hash:
            raise ValueError("Saved export hash differs from the checked source")
    except (OSError, ValueError):
        if created:
            destination.unlink(missing_ok=True)
        raise
    return destination


class VideoExportDialog(QDialog):
    """Observe an owned job without running native physics in the Qt process."""

    def __init__(
        self,
        source_fit_id: str,
        session: Any,
        parent: Any = None,
        *,
        library_root: Path,
    ) -> None:
        super().__init__(parent)
        self.source_fit_id, self.session = source_fit_id, session
        self.library_root = library_root
        self.run: dict[str, Any] | None = None
        self._worker: BackgroundWorker | None = None
        self._operation = ""
        self._cancel_requested = False
        self._closed = False
        self.setWindowTitle("Export Fitted Overlay")
        layout = QVBoxLayout(self)
        self.boundary = QLabel(
            f"Saved Fit: {source_fit_id}\nMonocular research hypothesis. Overlay follows original source presentation time; camera, anatomy and physical dynamics remain unqualified."
        )
        self.boundary.setWordWrap(True)
        layout.addWidget(self.boundary)
        self.force_layer = QCheckBox(
            "Draw force and torque glyphs (research fit; not measured forces)"
        )
        self.segment_shading = QCheckBox("Shade body segments")
        layout.addWidget(self.force_layer)
        layout.addWidget(self.segment_shading)
        self.start = QPushButton("Render Original-Footage Overlay")
        self.cancel = QPushButton("Cancel Export")
        self.save = QPushButton("Save Checked ZIP")
        self.cancel.setEnabled(False)
        self.save.setEnabled(False)
        for button, callback in (
            (self.start, self._start),
            (self.cancel, self._cancel),
            (self.save, self._save),
        ):
            button.clicked.connect(callback)
            layout.addWidget(button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

    def force_layer_settings(self) -> dict[str, Any] | None:
        """Opt-in layer settings, or None so the default export is unchanged."""
        if not self.force_layer.isChecked():
            return None
        return {
            "enabled": True,
            "kinds": ["joint_reaction"],
            "scale": 1.0,
            "segment_shading": self.segment_shading.isChecked(),
        }

    def _work(self, operation: str, target: Callable[[], Any]) -> None:
        def guarded() -> Any:
            try:
                return target()
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(str(exc)) from exc

        self._operation = operation
        self._worker = get_worker_adapter(guarded, force_threading=True)
        self.start.setEnabled(False)
        self.save.setEnabled(False)
        self._worker.start()
        self._timer.start()

    def _start(self) -> None:
        if self._worker or self._closed:
            return
        self.run = None
        self._cancel_requested = False
        self.cancel.setEnabled(True)
        self.status.setText("Submitting Research Overlay…")
        layer = self.force_layer_settings()
        self._work(
            "submit",
            lambda: (
                self.session.submit(self.source_fit_id, force_layer=layer)
                if layer
                else self.session.submit(self.source_fit_id)
            ),
        )

    def _poll(self) -> None:
        if self._closed:
            return
        if self._worker:
            if self._worker.is_running():
                return
            worker, self._worker = self._worker, None
            if worker.error:
                if self._operation == "save":
                    self._render()
                self.status.setText(str(worker.error))
                self._timer.stop()
                self.cancel.setEnabled(False)
                self.start.setEnabled(True)
                return
            if self._operation == "save":
                self._render()
                self.status.setText(f"Checked Research Overlay Saved: {worker.result}")
                self._timer.stop()
                return
            try:
                self._check_owner(worker.result)
            except ValueError as exc:
                self.status.setText(str(exc))
                self.start.setEnabled(True)
                self.cancel.setEnabled(False)
                self._timer.stop()
                return
            self.run = worker.result
        if self.run:
            try:
                updated = (
                    self.session.cancel(self.run["run_id"])
                    if self._cancel_requested and self.run["control_available"]
                    else self.session.view(self.run["run_id"])
                )
                self._check_owner(updated, self.run["run_id"])
                self.run = updated
                self._render()
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                self.save.setEnabled(False)
                self.status.setText(str(exc))

    def _check_owner(self, view: dict[str, Any], run_id: str | None = None) -> None:
        if view.get("source_fit_id") != self.source_fit_id or (
            run_id is not None and view.get("run_id") != run_id
        ):
            raise ValueError(
                "Export response does not belong to the selected fit and run"
            )

    def _render(self) -> None:
        if not self.run or self._closed:
            return
        self.status.setText(
            f"{self.run['status']} · {self.run['acceptance']} · {self.run['qualification']}\n{self.run['message']}\n"
            + "\n".join(self.run["blockers"])
        )
        active = self.run["status"] in {"pending", "running"}
        self.start.setEnabled(not active)
        self.cancel.setEnabled(active and bool(self.run["control_available"]))
        self.save.setEnabled(
            self.run["status"] == "succeeded"
            and self.run["download_available"]
            and self.run.get("execution_verified") is True
        )
        if not active:
            self._timer.stop()

    def _cancel(self) -> None:
        self._cancel_requested = True
        self._poll()

    def _save(self) -> None:
        if (
            self._closed
            or self._worker
            or not self.run
            or self.run["status"] != "succeeded"
            or not self.run["download_available"]
            or self.run.get("execution_verified") is not True
        ):
            return
        destination, _ = QFileDialog.getSaveFileName(
            self, "Save Research Overlay", "", "Research Overlay Package (*.zip)"
        )
        if destination:
            run_id = self.run["run_id"]
            self._work(
                "save",
                lambda: copy_export(
                    self.session.download(run_id), Path(destination), self.library_root
                ),
            )

    def reject(self) -> None:
        self._cancel()
        super().reject()

    def cleanup(self) -> None:
        """Drain owned copy/submission after the host closes its shared session."""
        if self._closed:
            return
        self._cancel()
        self._closed = True
        self._timer.stop()
        if self._worker:
            self._worker.wait()
            self._worker = None
