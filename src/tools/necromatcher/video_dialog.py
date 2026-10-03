"""Responsive native export controls over the shared isolated video service."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import os
import shutil
from typing import Any

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from src.shared.python.ui.adapters import BackgroundWorker, get_worker_adapter
from src.shared.python.workspace import compute_file_sha256
from src.shared.python.body_part_viz.overlay_options import ShapeOverlayOptions
from src.tools.necromatcher.refit_dialog import (
    ReviewedShaftDialog,
    read_reviewed_shaft_evidence,
    source_scope_summary,
)


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


class VideoExportDialog(ReviewedShaftDialog):
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
        self.stored_runs = QComboBox()
        self.stored_runs.setAccessibleName("Stored Overlay Exports")
        self.stored_runs.addItem("Select a Stored Overlay Export", None)
        layout.addWidget(self.stored_runs)
        self._build_shaft_inputs(layout)
        self._build_shape_inputs(layout)
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
        self.stored_runs.currentIndexChanged.connect(self._recall)
        self._load_stored_runs()

    def _build_shape_inputs(self, layout: QVBoxLayout) -> None:
        self.shape_enabled = QCheckBox("Show Translucent Model Proxy")
        self.shape_opacity = QDoubleSpinBox()
        self.shape_opacity.setAccessibleName("Model Proxy Opacity")
        self.shape_opacity.setRange(0.0, 1.0)
        self.shape_opacity.setSingleStep(0.05)
        self.shape_opacity.setValue(0.35)
        self.shape_opacity.setEnabled(False)
        self.shape_enabled.toggled.connect(self.shape_opacity.setEnabled)
        layout.addWidget(self.shape_enabled)
        layout.addWidget(self.shape_opacity)
        note = QLabel(
            "Model Proxy Retains the Skeleton; Authored Geometry Is Uncalibrated. "
            "Historical Anatomy and Original Scene Occlusion Remain Unknown."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

    def _shape_controls(self, blocked: bool) -> None:
        self.shape_enabled.setEnabled(not blocked)
        self.shape_opacity.setEnabled(not blocked and self.shape_enabled.isChecked())

    def _load_stored_runs(self) -> None:
        """List canonical fit-scoped persisted receipts without submitting work."""
        list_runs = getattr(self.session, "stored_runs", None)
        if not callable(list_runs):
            return
        try:
            records = list_runs(self.source_fit_id)
            for record in records:
                self._check_owner(record)
            for record in records:
                self.stored_runs.addItem(
                    f"{record['run_id']} · {record['status']} · {record['acceptance']}",
                    record["run_id"],
                )
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def _recall(self, index: int) -> None:
        """Observe a stored run through the authoritative session status boundary."""
        if self._closed or self._worker:
            return
        self._timer.stop()
        self.run = None
        self._cancel_requested = False
        self.save.setEnabled(False)
        self.cancel.setEnabled(False)
        self.start.setEnabled(True)
        run_id = self.stored_runs.itemData(index)
        if run_id is None:
            self.status.setText(
                "Select a Stored Export or Render a New Research Overlay."
            )
            return
        try:
            view = self._view_run(run_id)
            self._check_owner(view, run_id)
            self.run = view
            self._render()
            if view["status"] in {"pending", "running"}:
                self._timer.start()
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def _view_run(self, run_id: str) -> dict[str, Any]:
        """Use the fit/parent guard when supported by the canonical session."""
        guarded_view = getattr(self.session, "view_for_fit", None)
        if callable(guarded_view):
            return dict(guarded_view(self.source_fit_id, run_id))
        return dict(self.session.view(run_id))

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
        self.stored_runs.setEnabled(False)
        self._shaft_controls(True)
        self._shape_controls(True)
        self._worker.start()
        self._timer.start()

    def _start(self) -> None:
        if self._worker or self._closed:
            return
        if self._shaft_path is not None and self._shaft_owner != self.source_fit_id:
            self.status.setText("Remove evidence selected for another source fit")
            return
        source, path = self.source_fit_id, self._shaft_path
        shape = (
            ShapeOverlayOptions(self.shape_opacity.value())
            if self.shape_enabled.isChecked()
            else None
        )

        def submit() -> dict[str, Any]:
            evidence = read_reviewed_shaft_evidence(path)
            options = {"shape_overlay": shape} if shape is not None else {}
            return (
                self.session.submit(source, **options)
                if evidence is None
                else self.session.submit(source, evidence, **options)
            )

        self.run = None
        self._cancel_requested = False
        self.cancel.setEnabled(True)
        self.status.setText("Submitting Research Overlay…")
        self._work("submit", submit)

    def _poll(self) -> None:
        if self._closed:
            return
        if self._worker:
            if self._worker.is_running():
                return
            worker, self._worker = self._worker, None
            self.stored_runs.setEnabled(True)
            if worker.error:
                if self._operation == "save":
                    self._render()
                self.status.setText(str(worker.error))
                self._timer.stop()
                self.cancel.setEnabled(False)
                self.start.setEnabled(True)
                self._shaft_controls(False)
                self._shape_controls(False)
                return
            if self._operation == "save":
                self._render()
                self.status.setText(
                    self.status.text()
                    + f"\nChecked Research Overlay Saved: {worker.result}"
                )
                self._timer.stop()
                return
            try:
                self._check_owner(worker.result)
            except ValueError as exc:
                self.status.setText(str(exc))
                self.start.setEnabled(True)
                self._shaft_controls(False)
                self._shape_controls(False)
                self.cancel.setEnabled(False)
                self._timer.stop()
                return
            self.run = worker.result
        if self.run:
            try:
                updated = (
                    self.session.cancel(self.run["run_id"])
                    if self._cancel_requested and self.run["control_available"]
                    else self._view_run(self.run["run_id"])
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
        historical = (
            self.run["status"] == "succeeded"
            and self.run.get("execution_verified") is True
        )
        stored = historical and not self.run["download_available"]
        note = (
            "\nReadiness unverified; guarded verification can reject changed files."
            if stored
            else ""
        )
        producer = self.run.get("producer_source_commit")
        if producer:
            note += (
                f"\nProducer commit: {producer}; current source equality unverified."
            )
        if self.run.get("shape_overlay") is not None:
            shape = ShapeOverlayOptions.from_record(self.run["shape_overlay"])
            note += f"\nStored Model Proxy Opacity: {shape.opacity}; Skeleton Retained; Uncalibrated."
        if self.run.get("source_fit_scope") is not None:
            note += "\n" + source_scope_summary(
                self.run["source_fit_scope"], self.run.get("source_fit_scope_binding")
            )
        self.status.setText(
            f"{self.run['status']} · {self.run['acceptance']} · {self.run['qualification']}\n{self.run['message']}\n"
            + "\n".join(self.run["blockers"])
            + note
        )
        active = self.run["status"] in {"pending", "running"}
        self.start.setEnabled(not active)
        if hasattr(self, "shaft_import"):
            self._shaft_controls(active)
        if hasattr(self, "shape_enabled"):
            self._shape_controls(active)
        self.cancel.setEnabled(active and bool(self.run["control_available"]))
        self.save.setText(
            "Verify Stored Overlay Package" if stored else "Save Checked ZIP"
        )
        self.save.setEnabled(historical)
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
