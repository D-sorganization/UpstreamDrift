"""Native authored-replay impact controls over canonical background services."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZipFile

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from src.shared.python.ui.adapters import BackgroundWorker, get_worker_adapter
from src.shared.python.workspace import (
    ReplayImpactGeometry,
    ReplayImpactSelection,
    ShotTrajectoryHandoffCoordinator,
    compute_file_sha256,
    load_replay_impact_receipt,
)
from .video_dialog import copy_export


def read_impact_declaration(
    path: Path, sample_count: int
) -> tuple[ReplayImpactGeometry, ReplayImpactSelection]:
    """Use the public typed owners without inferring geometry or contact."""
    with path.open("rb") as incoming:
        raw = incoming.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Impact declaration exceeds 1 MiB")
    record = json.loads(raw)
    if not isinstance(record, dict) or set(record) != {"geometry", "selection"}:
        raise ValueError("Import exactly geometry and selection records")
    geometry = ReplayImpactGeometry.from_record(record["geometry"])
    selection = ReplayImpactSelection.from_record(record["selection"])
    if selection.recorded_sample_index >= sample_count:
        raise ValueError("Selected sample is outside the verified replay")
    return geometry, selection


def replay_summary(trace: Any) -> str:
    """Display canonical loaded parents and explicit authored-clock qualification."""
    meta = trace.meta
    flags = {
        "schema": "necromatcher/authored-replay/1",
        "scientific_qualified": False,
        "physical_source_time_qualified": False,
        "independent_replay_executed": True,
        "root_policy": "unactuated",
        "initial_state_policy": "exact_saved_pose_and_authored_rates",
    }
    if any(
        meta.get(key) is not value
        if isinstance(value, bool)
        else meta.get(key) != value
        for key, value in flags.items()
    ):
        raise ValueError("Replay qualification is missing or inconsistent")
    rows = []
    for parent in ("fit", "model", "profile", "capture"):
        identity, digest = meta.get(parent + "_id"), meta.get(parent + "_hash")
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or not isinstance(digest, str)
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest)
        ):
            raise ValueError("Replay parent identity/hash is invalid")
        rows.append(f"{parent}: {identity} · {digest}")
    if len(trace.t) < 2 or not math.isfinite(trace.dt) or trace.dt <= 0:
        raise ValueError("Replay sample clock is invalid")
    return (
        f"{len(trace.t)} Samples · Backend: {trace.backend} · Step: {trace.dt} authored s\n"
        "Scientific and physical source time unqualified; unactuated root; exact saved pose and authored rates.\n"
        + "\n".join(rows)
    )


def checked_impact_view(
    view: dict[str, Any], replay: str, sample_count: int, run_id: str | None = None
) -> dict[str, Any]:
    """Reject foreign controls and unverified summaries before display/export."""
    if not isinstance(view, dict):
        raise ValueError("Impact response must be an object")
    view = json.loads(json.dumps(view, allow_nan=False))
    if (
        view.get("replay_id") != replay
        or not isinstance(view.get("run_id"), str)
        or not re.fullmatch(r"[a-f0-9]{32}", view["run_id"])
        or (run_id is not None and view["run_id"] != run_id)
    ):
        raise ValueError("Impact response belongs to another replay/run")
    if (
        view.get("scientific_qualified") is not False
        or view.get("physical_source_time_qualified") is not False
    ):
        raise ValueError("Impact qualification is inconsistent")
    if view.get("status") not in {
        "pending",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    } or view.get("acceptance") not in {"partial", "interrupted", "rejected"}:
        raise ValueError("Impact status is invalid")
    for key in ("control_available", "execution_verified", "download_available"):
        if type(view.get(key)) is not bool:
            raise ValueError("Impact readiness requires explicit boolean flags")
    if (
        (view["execution_verified"] and view["status"] != "succeeded")
        or (view["control_available"] and view["status"] not in {"pending", "running"})
        or (view["status"] == "succeeded" and view["acceptance"] != "rejected")
    ):
        raise ValueError("Impact terminal/readiness flags are inconsistent")
    if (
        not isinstance(view.get("message"), str)
        or not isinstance(view.get("blockers"), list)
        or any(not isinstance(item, str) for item in view["blockers"])
    ):
        raise ValueError("Impact message/blockers must be explicit text")
    artifact = view.get("artifactsummary")
    if artifact is not None:
        if view["status"] != "succeeded" or not view["execution_verified"]:
            raise ValueError("Saved impact artifact is unverified")
        _check_saved_artifact(artifact, sample_count)
    if view["download_available"] and artifact is None:
        raise ValueError("Download requires a verified complete impact bundle")
    return view


def _check_saved_artifact(artifact: Any, sample_count: int) -> None:
    """Check the canonical saved declarations and finite retained metrics."""
    if (
        not isinstance(artifact, dict)
        or not isinstance(artifact.get("files"), list)
        or any(not isinstance(item, str) for item in artifact["files"])
        or sorted(artifact["files"])
        != sorted(
            [
                "trajectory.json",
                "impact-receipt.json",
                "result.json",
                "request.json",
            ]
        )
    ):
        raise ValueError("Impact bundle members differ")
    ReplayImpactGeometry.from_record(artifact["geometry"])
    selection = ReplayImpactSelection.from_record(artifact["selection"])
    if (
        selection.recorded_sample_index >= sample_count
        or artifact.get("clockpolicy") != "authored_simulation_seconds"
    ):
        raise ValueError("Saved impact selection/clock differs")
    summary = artifact["summary"]
    if (
        not isinstance(summary, dict)
        or set(summary)
        != {"carry_m", "max_height_m", "flight_time_s", "landing_angle_deg"}
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in summary.values()
        )
    ):
        raise ValueError("Saved impact metrics are invalid")


def load_verified_impact_curve(
    session: Any, replay_id: str, run_id: str
) -> tuple[Any, dict[str, Any]]:
    """Authenticate the fixed bundle and delegate retained-sample import publicly."""
    archive_path = session.download(replay_id, run_id)
    digest = compute_file_sha256(archive_path)
    expected = {"trajectory.json", "impact-receipt.json", "result.json", "request.json"}
    with TemporaryDirectory(prefix="necromatcher-impact-view-") as directory:
        root = Path(directory)
        with ZipFile(archive_path) as archive:
            if len(archive.namelist()) != 4 or set(archive.namelist()) != expected:
                raise ValueError(
                    "Impact viewer requires exactly the verified four ZIP members"
                )
            for name in expected:
                (root / name).write_bytes(archive.read(name))
        request = json.loads((root / "request.json").read_bytes())
        if (
            not isinstance(request, dict)
            or request.get("replay_id") != replay_id
            or request.get("run_id") != run_id
        ):
            raise ValueError("Impact viewer request belongs to another replay/run")
        state = load_replay_impact_receipt(
            root / "impact-receipt.json", root / "trajectory.json"
        )
        if (
            request.get("replay_id") != replay_id
            or state.metadata.get("replay_id") != replay_id
        ):
            raise ValueError("Impact viewer receipt belongs to another replay")
        if state.metadata.get("geometry") != request.get(
            "geometry"
        ) or state.metadata.get("selection") != request.get("selection"):
            raise ValueError("Impact viewer assumptions differ from the saved request")
        curve = ShotTrajectoryHandoffCoordinator(root).load_into_shot_tracer(
            root / "trajectory.json"
        )
        if compute_file_sha256(archive_path) != digest:
            raise ValueError("Impact ZIP changed during viewer handoff")
    return curve, dict(state.metadata)


class ReplayImpactDialog(QDialog):
    """Own a dedicated session; all authenticated I/O and shutdown stay off Qt."""

    def __init__(
        self, replay_id: str, session: Any, parent: Any = None, *, library_root: Path
    ) -> None:
        super().__init__(parent)
        self.replay_id, self.session, self.library_root = (
            replay_id,
            session,
            library_root,
        )
        self.sample_count = 0
        self.declaration: tuple[ReplayImpactGeometry, ReplayImpactSelection] | None = (
            None
        )
        self.run: dict[str, Any] | None = None
        self._worker: BackgroundWorker | None = None
        self._shutdown_worker: BackgroundWorker | None = None
        self._closed, self._cancel_requested = False, False
        self._operation = ""
        self.tracer_widget: Any = None
        self._tracer_windows: list[QDialog] = []
        self.setWindowTitle("Authored Replay Research Impact")
        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        self._work("load", lambda: self.session.library.load_replay(self.replay_id))

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        self.boundary = QLabel("Loading Verified Authored Replay…")
        self.boundary.setWordWrap(True)
        layout.addWidget(self.boundary)
        self.declaration_path = QLineEdit()
        self.declaration_path.setReadOnly(True)
        self.import_declaration = QPushButton("Import Impact Declaration JSON")
        self.import_declaration.clicked.connect(self._choose_declaration)
        self.budget = QLineEdit()
        self.budget.setAccessibleName("Impact Budget (s)")
        self.budget.setPlaceholderText(
            "Explicit wall budget in seconds: 0 < value ≤ 600"
        )
        self.budget.textChanged.connect(self._controls)
        self.saved_run = QLineEdit()
        self.saved_run.setAccessibleName("Saved Impact Run ID")
        self.saved_run.textChanged.connect(self._controls)
        self.run_id = QLineEdit()
        self.run_id.setReadOnly(True)
        self.run_id.setAccessibleName("Current Impact Run ID")
        self.start, self.recall = (
            QPushButton("Preview Research Impact"),
            QPushButton("Recall Research Impact Run"),
        )
        self.cancel, self.save = (
            QPushButton("Cancel Research Impact"),
            QPushButton("Save Checked Impact ZIP"),
        )
        self.open_tracer = QPushButton("Open in Shot Tracer")
        for widget in (
            self.declaration_path,
            self.import_declaration,
            self.budget,
            self.saved_run,
            self.run_id,
        ):
            layout.addWidget(widget)
        for button, callback in (
            (self.start, self._start),
            (self.recall, self._recall),
            (self.cancel, self._cancel),
            (self.save, self._save),
            (self.open_tracer, self._open_tracer),
        ):
            button.clicked.connect(callback)
            layout.addWidget(button)
        self.status = QLabel(
            "Authored seconds; scientific and physical source time unqualified. No contact event is inferred."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(
            QLabel(
                "Extract trajectory.json from the ZIP and use Shot Tracer’s Import Trajectory Record. Retain impact-receipt.json for assumptions and qualification."
            )
        )
        self._controls()

    def _controls(self) -> None:
        busy = self._closed or self._worker is not None
        active = bool(
            self.run
            and self.run["status"] in {"pending", "running"}
            and self.run["control_available"]
        )
        try:
            budget = float(self.budget.text())
            valid_budget = math.isfinite(budget) and 0 < budget <= 600
        except ValueError:
            valid_budget = False
        self.start.setEnabled(
            not busy and not active and self.declaration is not None and valid_budget
        )
        self.recall.setEnabled(
            not busy
            and not active
            and self.sample_count > 0
            and bool(re.fullmatch(r"[a-f0-9]{32}", self.saved_run.text()))
        )
        self.import_declaration.setEnabled(
            not busy and not active and self.sample_count > 0
        )
        self.budget.setEnabled(not busy and not active)
        self.saved_run.setEnabled(not busy and not active)
        self.cancel.setEnabled(
            not self._closed
            and bool(active and self.run and self.run["control_available"])
        )
        self.save.setEnabled(
            not busy
            and bool(
                self.run
                and self.run["status"] == "succeeded"
                and self.run["execution_verified"]
                and self.run["download_available"]
            )
        )
        self.open_tracer.setEnabled(self.save.isEnabled())

    def _work(self, operation: str, target: Callable[[], Any]) -> None:
        if self._closed or self._worker:
            return
        self._operation = operation
        self._worker = get_worker_adapter(target, force_threading=True)
        self._controls()
        self._worker.start()
        self._timer.start()

    def _choose_declaration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Impact Declaration", "", "JSON (*.json)"
        )
        if path:
            self.load_declaration(Path(path))

    def load_declaration(self, path: Path) -> None:
        """Load exact public DTOs in a background operation without GUI math."""
        if self._closed or self._worker:
            return
        self.declaration = None
        self.declaration_path.setText(str(path))
        self._work(
            "declaration", lambda: read_impact_declaration(path, self.sample_count)
        )

    def _start(self) -> None:
        if not self.start.isEnabled() or self.declaration is None:
            return
        geometry, selection = self.declaration
        budget = float(self.budget.text())
        self.run = None
        self.run_id.clear()
        self.status.setText("Submitting Research Impact…")
        self._cancel_requested = False
        self._work(
            "submit",
            lambda: self.session.submit(self.replay_id, geometry, selection, budget),
        )

    def _recall(self) -> None:
        if not self.recall.isEnabled():
            return
        run_id = self.saved_run.text()
        self.run = None
        self.run_id.clear()
        self.status.setText("Recalling Verified Saved Impact…")
        self._work(
            "recall",
            lambda: checked_impact_view(
                self.session.view(self.replay_id, run_id),
                self.replay_id,
                self.sample_count,
                run_id,
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
                if self._operation != "save" or not isinstance(
                    worker.error, FileExistsError
                ):
                    self.run = None
                    self.run_id.clear()
                self.status.setText(str(worker.error))
                self._timer.stop()
                self._controls()
                return
            try:
                self._completed(worker.result)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                self.status.setText(str(exc))
                self.run = None
            self._controls()
        if (
            self.run
            and self.run["status"] in {"pending", "running"}
            and self.run["control_available"]
        ):
            run_id = self.run["run_id"]
            operation = "cancel" if self._cancel_requested else "view"
            self._cancel_requested = False
            method = self.session.cancel if operation == "cancel" else self.session.view
            self._work(
                operation,
                lambda: checked_impact_view(
                    method(self.replay_id, run_id),
                    self.replay_id,
                    self.sample_count,
                    run_id,
                ),
            )
        else:
            self._timer.stop()

    def _completed(self, result: Any) -> None:
        if self._operation == "load":
            self.boundary.setText(
                f"Replay: {self.replay_id}\n" + replay_summary(result)
            )
            self.sample_count = len(result.t)
        elif self._operation == "declaration":
            self.declaration = result
            self.status.setText(
                f"Declared Sample: {result[1].recorded_sample_index}\n{result[0].assumption_description}\n{result[1].selection_description}"
            )
        elif self._operation == "save":
            self.status.setText(f"Checked Research Impact ZIP Saved: {result}")
        elif self._operation == "tracer":
            self._show_tracer(*result)
        else:
            self.run = checked_impact_view(result, self.replay_id, self.sample_count)
            self._render()

    def _render(self) -> None:
        if not self.run:
            return
        self.run_id.setText(self.run["run_id"])
        text = (
            f"{self.run['status']} · {self.run['acceptance']}\n{self.run['message']}\n"
            + "\n".join(self.run["blockers"])
        )
        artifact = self.run.get("artifactsummary")
        if artifact:
            geometry, selection, summary = (
                artifact["geometry"],
                artifact["selection"],
                artifact["summary"],
            )
            text += (
                f"\nSample: {selection['recorded_sample_index']} · Clock: {artifact['clockpolicy']}"
                f"\n{geometry['assumption_description']}\n{selection['selection_description']}"
                f"\nBody: {geometry['body']} · Mass: {geometry['mass_kg']} kg · MOI: {geometry['moi_kg_m2']} kg·m²"
                f"\nCarry: {summary['carry_m']} m · Height: {summary['max_height_m']} m · Flight: {summary['flight_time_s']} s · Landing: {summary['landing_angle_deg']}°"
            )
        elif (
            self.run["status"] in {"pending", "running"}
            and not self.run["control_available"]
        ):
            text += "\nReadiness unverified; this host has no live control handle. Recall another saved Run ID if needed."
        self.status.setText(text)

    def _cancel(self) -> None:
        if self.cancel.isEnabled():
            self._cancel_requested = True
            self._poll()

    def _save(self) -> None:
        if not self.save.isEnabled():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Research Impact Bundle", "", "ZIP (*.zip)"
        )
        if path:
            self._save_to(Path(path))

    def _save_to(self, destination: Path) -> None:
        if not self.save.isEnabled() or self.run is None:
            return
        run_id = self.run["run_id"]
        self._work(
            "save",
            lambda: copy_export(
                self.session.download(self.replay_id, run_id),
                destination,
                self.library_root,
            ),
        )

    def _open_tracer(self) -> None:
        if not self.open_tracer.isEnabled() or self.run is None:
            return
        run_id = self.run["run_id"]
        self._work(
            "tracer",
            lambda: load_verified_impact_curve(self.session, self.replay_id, run_id),
        )

    def _show_tracer(self, curve: Any, metadata: dict[str, Any]) -> None:
        from src.launchers.shot_tracer import MultiModelShotTracerWidget

        host = QDialog(self)
        host.setWindowTitle("Retained Research Impact Trajectory")
        layout = QVBoxLayout(host)
        note = QLabel(
            f"Replay: {metadata['replay_id']} · Sample: {metadata['recorded_sample_index']}\nAuthored seconds; scientific and physical source time unqualified. Retained trajectory, not re-simulation."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        widget = MultiModelShotTracerWidget(host)
        widget.display_imported_trajectory(curve)
        layout.addWidget(widget)
        host.resize(1000, 700)
        self.tracer_widget = widget
        self._tracer_windows.append(host)
        host.show()

    def cleanup(self) -> None:
        """Cancel/drain the dedicated session off Qt; late results remain hidden."""
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._controls()
        worker = self._worker

        def drain() -> None:
            try:
                if worker:
                    worker.wait()
            finally:
                self.session.close()
                self._worker = None

        self._shutdown_worker = get_worker_adapter(drain, force_threading=True)
        self._shutdown_worker.start()

    def reject(self) -> None:
        self.cleanup()
        super().reject()
