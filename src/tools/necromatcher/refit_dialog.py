"""Responsive desktop research refits using the shared matching session."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from src.shared.python.motion_matching.historical_fit import ImageFitConfig
from src.shared.python.ui.adapters import BackgroundWorker, get_worker_adapter
from src.shared.python.workspace import NativeRefitOptions, NativeRefitSession


class ResearchRefitDialog(QDialog):
    """Own a source-scoped form and observer; execution stays in the shared service."""

    stored = pyqtSignal()

    def __init__(
        self,
        source_fit_id: str,
        plan: dict[str, Any],
        session: NativeRefitSession,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.source_fit_id = source_fit_id
        self.session = session
        self.run: dict[str, Any] | None = None
        self._worker: BackgroundWorker | None = None
        self._cancel_requested = False
        self._notified = False
        self.setWindowTitle("Research Refit")
        layout = QVBoxLayout(self)
        label = QLabel(
            f"Source Version: {source_fit_id}\nNew versions preserve the original. Physical time, camera and dynamics remain unqualified."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        form = QFormLayout()
        self.identity = QLineEdit()
        self.frames = QLineEdit()
        self.scales = QLineEdit()
        previous = plan.get("recorded_options") or {}
        indices = previous.get("frame_indices", plan["frame_indices"])
        if not previous:
            step = max(1, (len(indices) + 59) // 60)
            indices = sorted(set(indices[::step] + [indices[-1]]))
        self.frames.setText(", ".join(map(str, indices)))
        self.scales.setText(", ".join(map(str, previous.get("coordinate_scales", []))))
        form.addRow("New Fit Version", self.identity)
        form.addRow("Source Frame Indices", self.frames)
        coordinates = QLabel(
            ", ".join(
                f"{name} ({unit})"
                for name, unit in zip(
                    plan["coordinate_order"], plan["coordinate_units"], strict=True
                )
            )
        )
        coordinates.setWordWrap(True)
        form.addRow("Coordinate Order", coordinates)
        form.addRow("Coordinate Prior Scales", self.scales)
        defaults = {
            "knot_count": min(12, len(indices)),
            **asdict(ImageFitConfig()),
            "budget_wall_s": 600.0,
            "unknown_visibility_weight": 0.5,
        }
        defaults.update(
            {key: value for key, value in previous.items() if key in defaults}
        )
        defaults.update(previous.get("config", {}))
        self.fields: dict[str, QLineEdit] = {}
        labels = {
            "knot_count": "Spline Knots",
            "max_iterations": "Evaluation Budget",
            "prior_weight": "Pose Prior Weight",
            "smoothness_weight": "Smoothness Weight",
            "closure_weight": "Grip Closure Weight",
            "budget_wall_s": "Wall Budget (Seconds)",
            "unknown_visibility_weight": "Unknown Visibility Weight",
        }
        for name, title in labels.items():
            self.fields[name] = QLineEdit(str(defaults[name]))
            form.addRow(title, self.fields[name])
        layout.addLayout(form)
        self.start = QPushButton("Start Research Refit")
        self.start.clicked.connect(self._start)
        self.cancel = QPushButton("Cancel Research Refit")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self._cancel)
        layout.addWidget(self.start)
        layout.addWidget(self.cancel)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)

    def _options(self) -> NativeRefitOptions:
        values = {name: field.text() for name, field in self.fields.items()}
        return NativeRefitOptions(
            tuple(int(x.strip()) for x in self.frames.text().split(",")),
            int(values["knot_count"]),
            tuple(float(x.strip()) for x in self.scales.text().split(",")),
            ImageFitConfig(
                int(values["max_iterations"]),
                float(values["prior_weight"]),
                float(values["smoothness_weight"]),
                float(values["closure_weight"]),
            ),
            float(values["unknown_visibility_weight"]),
            float(values["budget_wall_s"]),
        )

    def _start(self) -> None:
        try:
            options = self._options()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        identity = self.identity.text().strip()
        self._worker = get_worker_adapter(
            lambda: self.session.submit(self.source_fit_id, identity, options),
            force_threading=True,
        )
        self._cancel_requested = False
        self._notified = False
        self.run = None
        self.start.setEnabled(False)
        self.cancel.setEnabled(True)
        self.status.setText("Submitting Research Refit…")
        self._worker.start()
        self._timer.start()

    def _render(self) -> None:
        if self.run is None:
            return
        self.status.setText(
            f"{self.run['status']} · {self.run['acceptance']} · {self.run['message']}\nRun: {self.run['run_id']}\n"
            + "\n".join(self.run["blockers"])
        )
        active = self.run["status"] in {"running", "pending"}
        self.start.setEnabled(not active)
        self.cancel.setEnabled(active)
        if not active:
            self._timer.stop()
        if self.run["status"] == "succeeded" and not self._notified:
            self._notified = True
            self.stored.emit()

    def _poll(self) -> None:
        if self._worker:
            if self._worker.is_running():
                return
            worker, self._worker = self._worker, None
            if worker.error:
                self.status.setText(str(worker.error))
                self.start.setEnabled(True)
                self.cancel.setEnabled(False)
                self._timer.stop()
                return
            self.run = worker.result
        if self.run:
            try:
                self.run = (
                    self.session.cancel(self.run["run_id"])
                    if self._cancel_requested
                    else self.session.view(self.run["run_id"])
                )
                self._render()
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                self.status.setText(str(exc))

    def _cancel(self) -> None:
        self._cancel_requested = True
        self._poll()

    def reject(self) -> None:
        self._cancel()
        super().reject()

    def cleanup(self) -> None:
        """Drain submission after the host cancels/closes its shared session."""
        self._cancel()
        self._timer.stop()
        if self._worker:
            self._worker.wait()
            self._worker = None
