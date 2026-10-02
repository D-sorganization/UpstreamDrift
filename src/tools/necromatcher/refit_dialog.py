"""Responsive desktop research refits using the shared matching session."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, cast

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QComboBox,
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
        layout.addLayout(self._build_form(plan))
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

    def _build_form(self, plan: dict[str, Any]) -> QFormLayout:
        form = QFormLayout()
        self.identity = QLineEdit()
        self.frames = QLineEdit()
        self.scales = QLineEdit()
        previous = plan.get("recorded_options") or {}
        self._preserved = plan.get("preserved_spline") or {}
        self._source_indices = tuple(plan["frame_indices"])
        record = plan.get("baseline_config")
        if record is None:
            record = previous.get("config", {})
        self._baseline_config = (
            record
            if isinstance(record, ImageFitConfig)
            else ImageFitConfig.from_record(record)
        )
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
            **asdict(self._baseline_config),
            "budget_wall_s": 600.0,
            "unknown_visibility_weight": 0.5,
        }
        defaults.update(
            {key: value for key, value in previous.items() if key in defaults}
        )
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
        self._sample_knot_count = self.fields["knot_count"].text()
        self._sample_frames = self.frames.text()
        self.initialization = QComboBox()
        self.initialization.addItem("Sample Parent Poses", "sampled_parent")
        self.initialization.addItem("Resume Saved Spline", "preserved_spline")
        model = cast(QStandardItemModel, self.initialization.model())
        item = model.item(1)
        if item is not None:
            item.setEnabled(self._preserved.get("available") is True)
        self.initialization.currentIndexChanged.connect(self._initialization_changed)
        form.addRow("Initialization", self.initialization)
        self.recipe_summary = QLabel()
        self.recipe_summary.setWordWrap(True)
        form.addRow("Retained Recipe", self.recipe_summary)
        self._update_recipe_summary()
        return form

    def _initialization_changed(self, index: int) -> None:
        exact = index == 1
        field = self.fields["knot_count"]
        if exact:
            self._sample_knot_count = field.text()
            self._sample_frames = self.frames.text()
            try:
                self.frames.setText(", ".join(map(str, self._resume_indices())))
            except ValueError:
                pass  # Submission validates malformed selected sample IDs.
            field.setText(str(self._preserved.get("knot_count", "")))
        else:
            field.setText(self._sample_knot_count)
            self.frames.setText(self._sample_frames)
        field.setEnabled(not exact)
        self.frames.setEnabled(not exact)
        self._update_recipe_summary()

    def _resume_indices(self) -> tuple[int, ...]:
        selected = {int(x.strip()) for x in self._sample_frames.split(",")}
        return tuple(
            sorted(selected | {self._source_indices[0], self._source_indices[-1]})
        )

    def _update_recipe_summary(self) -> None:
        config = self._baseline_config
        constraints = config.constraint_options
        pins = constraints.pinned_spheres if constraints else ()
        policy = (
            "strict"
            if self.initialization.currentIndex() == 1
            else config.initialization_policy
        )
        resume = (
            f"Saved Spline: {self._preserved.get('knot_count')} Knots, "
            f"Source Interval {self._preserved.get('source_interval')}"
            if self._preserved.get("available") is True
            else f"Resume Unavailable: {self._preserved.get('reason', 'No saved spline')}"
        )
        self.recipe_summary.setText(
            f"{len(config.coordinate_bounds)} Authored Bounds; "
            f"{len(config.interior_fractions)} Interior Fractions "
            f"{config.interior_fractions}; Pins: {', '.join(pins) or 'None'}; "
            f"Initialization Policy: {policy}. {resume}. "
            "Grip, Contact, Camera and Physical Time Remain Unqualified."
        )

    def _options(self) -> NativeRefitOptions:
        values = {name: field.text() for name, field in self.fields.items()}
        exact = self.initialization.currentIndex() == 1
        if exact and self._preserved.get("available") is not True:
            raise ValueError(self._preserved.get("reason", "No saved spline"))
        config = replace(
            self._baseline_config,
            max_iterations=int(values["max_iterations"]),
            prior_weight=float(values["prior_weight"]),
            smoothness_weight=float(values["smoothness_weight"]),
            closure_weight=float(values["closure_weight"]),
            initialization_policy=(
                "strict" if exact else self._baseline_config.initialization_policy
            ),
        )
        indices = (
            self._resume_indices()
            if exact
            else tuple(int(x.strip()) for x in self.frames.text().split(","))
        )
        return NativeRefitOptions(
            frame_indices=indices,
            knot_count=(
                self._preserved["knot_count"] if exact else int(values["knot_count"])
            ),
            coordinate_scales=tuple(
                float(x.strip()) for x in self.scales.text().split(",")
            ),
            config=config,
            unknown_visibility_weight=float(values["unknown_visibility_weight"]),
            budget_wall_s=float(values["budget_wall_s"]),
            operation="fit",
            initialization_source="preserved_spline" if exact else "sampled_parent",
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
