"""Explicit local research lifecycle for authenticated saved impact shots."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.golf_simulator import (
    GolfSessionService,
    LocalReferenceAdapter,
    SessionState,
    SubmissionState,
)
from src.shared.python.ui.adapters import BackgroundWorker, get_worker_adapter
from src.shared.python.workspace import ResearchImpactShot


class ResearchGolfDialog(QDialog):
    """Use actual local service controls; no external destination or simulated GUI tokens."""

    def __init__(self, parent: QWidget | None, admitted: ResearchImpactShot) -> None:
        if not isinstance(admitted, ResearchImpactShot):
            raise TypeError(
                "Research dialog requires an authenticated ResearchImpactShot"
            )
        super().__init__(parent)
        self.admitted = admitted
        self.service: GolfSessionService | None = None
        self.adapter: LocalReferenceAdapter | None = None
        self._worker: BackgroundWorker | None = None
        self._shutdown_worker: BackgroundWorker | None = None
        self._operation = ""
        self._closed = self._submitted = False
        self._attempted = False
        self._prepared_id: str | None = None
        self._token: str | None = None
        self.setWindowTitle("Local Research Simulation")
        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        self._controls()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        self.boundary = QLabel(
            "MODEL_CONTACT · Scientific, numerical and contact qualification UNVERIFIED.\n"
            "Authored seconds; declared source-to-target rotation is retained in the read-only context.\n"
            "This new local flight uses the local adapter default environment, not the retained original flight.\n"
            "No native avatar animation or course feedback capability."
        )
        self.boundary.setWordWrap(True)
        layout.addWidget(self.boundary)
        context = QPlainTextEdit(
            json.dumps(self.admitted.to_record(), indent=2, allow_nan=False)
        )
        context.setReadOnly(True)
        context.setMaximumHeight(180)
        context.setAccessibleName("Saved Research Assumptions and Provenance")
        layout.addWidget(context)
        for name, title, operation in (
            ("connect_button", "Connect Local Simulator", "connect"),
            ("prepare_button", "Prepare Research Shot", "prepare"),
            ("arm_button", "Arm Research Shot", "arm"),
            ("submit_button", "Submit Local Research Shot", "submit"),
            ("disarm_button", "Disarm", "disarm"),
            ("cancel_button", "Cancel Prepared Shot", "cancel"),
            ("recall_button", "Recall Retained Local Samples", "recall"),
        ):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, op=operation: self._work(op))
            setattr(self, name, button)
            layout.addWidget(button)
        self.status = QLabel(
            "Disconnected · Explicit Connect → Prepare → Arm → Submit Required"
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.samples = QTableWidget(0, 7)
        self.samples.setHorizontalHeaderLabels(
            ["t (s)", "x (m)", "y (m)", "z (m)", "vx (m/s)", "vy (m/s)", "vz (m/s)"]
        )
        layout.addWidget(self.samples)
        self.resize(850, 700)

    def _controls(self) -> None:
        idle = not self._closed and self._worker is None
        state = self.service.current_state if self.service else None
        self.connect_button.setEnabled(idle and self.service is None)
        self.prepare_button.setEnabled(
            idle and state == SessionState.IDLE and not self._attempted
        )
        self.arm_button.setEnabled(idle and state == SessionState.PREPARED)
        self.submit_button.setEnabled(idle and state == SessionState.ARMED)
        self.disarm_button.setEnabled(idle and state == SessionState.ARMED)
        self.cancel_button.setEnabled(
            idle and state in (SessionState.PREPARED, SessionState.ARMED)
        )
        self.recall_button.setEnabled(idle and self._submitted)

    def _work(self, operation: str) -> None:
        buttons = {
            "connect": self.connect_button,
            "prepare": self.prepare_button,
            "arm": self.arm_button,
            "submit": self.submit_button,
            "disarm": self.disarm_button,
            "cancel": self.cancel_button,
            "recall": self.recall_button,
        }
        if operation not in buttons or not buttons[operation].isEnabled():
            return
        self._operation = operation
        if operation == "submit":
            self._attempted = True
        self.status.setText(f"Local Research Operation: {operation.title()}…")
        self._worker = get_worker_adapter(
            lambda: asyncio.run(self._execute(operation)), force_threading=True
        )
        self._controls()
        self._worker.start()
        self._timer.start()

    async def _execute(self, operation: str) -> Any:
        if operation == "connect":
            adapter = LocalReferenceAdapter()
            connected_service = GolfSessionService(
                self.admitted.shot.session_id, adapter
            )
            status = await connected_service.select_destination(adapter)
            self.adapter, self.service = adapter, connected_service
            return status
        service = self.service
        if service is None:
            raise RuntimeError("Connect the actual local simulator first")
        if operation == "prepare":
            return service.prepare_research_shot(self.admitted.shot)
        if operation == "recall":
            return service.get_local_trajectory_record(self.admitted.shot.shot_id)
        if self._prepared_id is None:
            raise RuntimeError("No public prepared shot identity")
        if operation == "arm":
            return service.arm(self._prepared_id, 1)
        if operation == "disarm":
            return service.disarm(self._prepared_id)
        if operation == "cancel":
            return service.cancel(self._prepared_id)
        if operation == "submit" and self._token is not None:
            receipt = await service.submit_at_impact(self._prepared_id, self._token)
            record = (
                service.get_local_trajectory_record(receipt.shot_id)
                if receipt.state == SubmissionState.CONFIRMED_ACCEPTED
                else None
            )
            return receipt, record
        raise RuntimeError("Missing public arm token or unsupported research operation")

    def _poll(self) -> None:
        if self._closed or self._worker is None or self._worker.is_running():
            return
        worker, self._worker = self._worker, None
        self._timer.stop()
        if worker.error:
            self.status.setText(str(worker.error))
        else:
            try:
                self._apply(worker.result)
            except (ValueError, TypeError, AttributeError) as exc:
                self.status.setText(str(exc))
        self._controls()

    def _apply(self, result: Any) -> None:
        operation = self._operation
        if operation == "connect":
            self.status.setText(f"Connected: {result.endpoint}")
        elif operation == "prepare":
            self._prepared_id = result.prepared_shot_id
            self.status.setText(
                "Research Shot Prepared · All Qualifications Unverified"
            )
        elif operation == "arm":
            self._token = result
            self.status.setText("Armed · One Local Submission Available")
        elif operation in ("cancel", "disarm"):
            self._token = None
            if operation == "cancel":
                self._prepared_id = None
            self.status.setText(f"Research Shot {operation.title()}ed")
        elif operation == "submit":
            receipt, record = result
            self._submitted = receipt.state == SubmissionState.CONFIRMED_ACCEPTED
            self._token = None
            self.status.setText(
                f"Delivery: {receipt.state.value} · {receipt.detail}\nScientific Acceptance Remains Unverified"
            )
            if record is not None:
                self._show_samples(record)
        elif operation == "recall":
            self._show_samples(result)

    def _show_samples(self, record: Any) -> None:
        if record.shot_id != self.admitted.shot.shot_id:
            raise ValueError("Retained local samples belong to another shot")
        self.samples.setRowCount(len(record.points))
        for row, point in enumerate(record.points):
            for column, value in enumerate(
                (point.time, *point.position, *point.velocity)
            ):
                self.samples.setItem(row, column, QTableWidgetItem(repr(float(value))))
        self.status.setText(
            self.status.text()
            + f"\nRetained Local Samples: {len(record.points)} · No Re-simulation"
        )

    def cleanup(self) -> None:
        """Drain in a background worker; never publish late callbacks after closing."""
        if self._closed:
            return
        self._closed = True
        self._token = None
        self._timer.stop()
        self._controls()
        worker = self._worker
        prepared_id = self._prepared_id
        operation = self._operation

        def drain() -> None:
            try:
                if worker:
                    worker.wait()
                owned_id = prepared_id
                if worker and not worker.error and operation == "prepare":
                    owned_id = worker.result.prepared_shot_id
                if (
                    self.service
                    and self.service.current_state
                    in (SessionState.PREPARED, SessionState.ARMED)
                    and owned_id is not None
                ):
                    self.service.cancel(owned_id)
            finally:
                if self.adapter:
                    asyncio.run(self.adapter.disconnect())

        self._shutdown_worker = get_worker_adapter(drain, force_threading=True)
        self._shutdown_worker.start()

    def reject(self) -> None:
        self.cleanup()
        super().reject()
