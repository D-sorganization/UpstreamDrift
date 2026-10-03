"""Responsive desktop research refits using the shared matching session."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, cast

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QFileDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from src.shared.python.motion_matching.historical_fit import (
    ImageFitConfig,
    ScheduledConstraintOptions,
    ShaftAxisEvidence,
)
from src.shared.python.ui.adapters import BackgroundWorker, get_worker_adapter
from src.shared.python.workspace import (
    NativeRefitOptions,
    NativeRefitSession,
    SourceFitScope,
    import_fit_source_scope_review,
)
from src.shared.python.workspace.necromatcher_scope_import import (
    MAX_SOURCE_SCOPE_REVIEW_BYTES,
)


def read_reviewed_source_scope(
    path: Path | None,
    session: NativeRefitSession | None = None,
    fit_id: str | None = None,
) -> SourceFitScope | None:
    """Read a typed declaration off the Qt thread; the session authenticates it."""
    if path is None:
        return None
    if path.stat().st_size > MAX_SOURCE_SCOPE_REVIEW_BYTES:
        raise ValueError("Reviewed fitting window requires nonempty bytes up to 1 MiB")
    with path.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_SCOPE_REVIEW_BYTES + 1)
    if not raw or len(raw) > MAX_SOURCE_SCOPE_REVIEW_BYTES:
        raise ValueError("Reviewed fitting window requires nonempty bytes up to 1 MiB")
    record = json.loads(raw)
    if not isinstance(record, dict):
        raise ValueError("Reviewed fitting window must be a JSON object")
    if record.get("schema") == "necromatcher/source-fit-scope-review/1":
        if session is None or fit_id is None:
            raise ValueError(
                "Raw reviewed window requires the selected fit and library"
            )
        return import_fit_source_scope_review(session.library, fit_id, raw)
    return SourceFitScope.from_record(record)


def source_scope_summary(
    record: dict[str, Any], binding: dict[str, Any] | None = None
) -> str:
    """Describe reviewed bounds separately from the stored selected source domain."""
    scope = SourceFitScope.from_record(record)
    result = (
        f"Reviewed Original Frames: {scope.first_frame} to "
        f"{scope.end_exclusive_frame} (Exclusive). {scope.review.reason}. "
        f"{scope.review.uncertainty_policy}. Hand contact and physical release remain unmeasured."
    )
    if binding is not None:
        indices = binding["frame_indices"]
        first = "/".join(map(str, binding["first_pts"]))
        last = "/".join(map(str, binding["last_pts"]))
        result += f" Stored Fit Domain: Frames {indices[0]} to {indices[-1]}; Source PTS {first} to {last}. Physical time unqualified."
    return result


def read_reviewed_shaft_evidence(path: Path | None) -> ShaftAxisEvidence | None:
    """Read typed evidence in a worker; canonical session admission rebinds sources."""
    if path is None:
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Reviewed shaft evidence must be a JSON object")
    return ShaftAxisEvidence.from_record(record)


class ReviewedShaftDialog(QDialog):
    """Source-owned optional evidence controls shared by refit and export forms."""

    source_fit_id: str

    def _build_shaft_inputs(self, layout: QVBoxLayout) -> None:
        self._shaft_path: Path | None = None
        self._shaft_owner: str | None = None
        self.shaft_import = QPushButton("Choose Reviewed Shaft JSON")
        self.shaft_remove = QPushButton("Remove Shaft Evidence")
        self.shaft_remove.setEnabled(False)
        self.shaft_status = QLabel()
        self.shaft_status.setWordWrap(True)
        self.shaft_import.clicked.connect(self._choose_shaft)
        self.shaft_remove.clicked.connect(self._remove_shaft)
        for widget in (self.shaft_import, self.shaft_remove, self.shaft_status):
            layout.addWidget(widget)
        self._remove_shaft()

    def _choose_shaft(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Reviewed Shaft Evidence", "", "Reviewed Evidence (*.json)"
        )
        if path:
            self._shaft_path = Path(path)
            self._shaft_owner = self.source_fit_id
            self.shaft_status.setText(
                f"Selected: {self._shaft_path.name}; validation pending. "
                "Confidence and uncertainty are authored, uncalibrated; no physical endpoint claims."
            )
            self.shaft_remove.setEnabled(True)

    def _remove_shaft(self) -> None:
        self._shaft_path = None
        self._shaft_owner = None
        self.shaft_remove.setEnabled(False)
        self.shaft_status.setText(
            "Shaft evidence disabled. Optional reviewed image fragments; "
            "authored, uncalibrated confidence and uncertainty."
        )

    def _shaft_controls(self, active: bool) -> None:
        self.shaft_import.setEnabled(not active)
        self.shaft_remove.setEnabled(not active and self._shaft_path is not None)


class ResearchRefitDialog(ReviewedShaftDialog):
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
        self._build_shaft_inputs(layout)
        self._build_scope_inputs(layout, plan)
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

    def _build_scope_inputs(self, layout: QVBoxLayout, plan: dict[str, Any]) -> None:
        self._scope_path: Path | None = None
        self._scope_owner: str | None = None
        self._inherited_scope = plan.get("source_scope")
        self._inherited_scope_binding = plan.get("source_scope_binding")
        self.scope_import = QPushButton("Import Reviewed Window")
        self.scope_remove = QPushButton("Remove Imported Window")
        self.scope_status = QLabel()
        self.scope_status.setWordWrap(True)
        self.scope_import.clicked.connect(self._choose_scope)
        self.scope_remove.clicked.connect(self._remove_scope)
        for widget in (self.scope_import, self.scope_remove, self.scope_status):
            layout.addWidget(widget)
        self._remove_scope()

    def _choose_scope(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Reviewed Window", "", "Reviewed Window (*.json)"
        )
        if path:
            self._scope_path = Path(path)
            self._scope_owner = self.source_fit_id
            self.scope_remove.setEnabled(True)
            self.scope_status.setText(
                f"Selected: {self._scope_path.name}; source and review validation pending. "
                "Hand contact and physical release remain unmeasured."
            )

    def _remove_scope(self) -> None:
        self._scope_path = None
        self._scope_owner = None
        self.scope_remove.setEnabled(False)
        scope = self._inherited_scope
        if scope is None:
            self.scope_status.setText(
                "No reviewed window imported; inherited scope is retained."
            )
            return
        self.scope_status.setText(
            source_scope_summary(scope, self._inherited_scope_binding)
        )

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
        schedule_summary = ""
        if isinstance(constraints, ScheduledConstraintOptions):
            phases = constraints.schedule.phases
            pins = tuple(
                sorted({pin for phase in phases for pin in phase.pinned_spheres})
            )
            boundaries = constraints.schedule.boundary_times()
            start, end = boundaries[0], boundaries[-1]
            schedule_summary = (
                f"{len(phases)} Authored Contact Phases; "
                f"Review Source Interval: {start:g} to {end:g}; "
                "Authored Contact Hypothesis. "
            )
        else:
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
            f"Initialization Policy: {policy}. {resume}. {schedule_summary}"
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
        if self._worker:
            return
        if self._shaft_path is not None and self._shaft_owner != self.source_fit_id:
            self.status.setText("Remove evidence selected for another source fit")
            return
        if self._scope_path is not None and self._scope_owner != self.source_fit_id:
            self.status.setText("Remove window selected for another source fit")
            return
        try:
            options = self._options()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        identity = self.identity.text().strip()
        source, path = self.source_fit_id, self._shaft_path
        scope_path = self._scope_path

        def submit() -> dict[str, Any]:
            evidence = read_reviewed_shaft_evidence(path)
            scope = read_reviewed_source_scope(scope_path, self.session, source)
            if evidence is None:
                if scope is None:
                    return self.session.submit(source, identity, options)
                return self.session.submit(
                    source, identity, options, source_scope=scope
                )
            if scope is None:
                return self.session.submit(source, identity, options, evidence)
            return self.session.submit(
                source, identity, options, evidence, source_scope=scope
            )

        self._worker = get_worker_adapter(
            submit,
            force_threading=True,
        )
        self._cancel_requested = False
        self._notified = False
        self.run = None
        self.start.setEnabled(False)
        self._shaft_controls(True)
        self.scope_import.setEnabled(False)
        self.scope_remove.setEnabled(False)
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
        if self.run.get("source_fit_scope") is not None:
            self.scope_status.setText(
                source_scope_summary(
                    self.run["source_fit_scope"],
                    self.run.get("source_fit_scope_binding"),
                )
            )
        active = self.run["status"] in {"running", "pending"}
        self.start.setEnabled(not active)
        self._shaft_controls(active)
        self.scope_import.setEnabled(not active)
        self.scope_remove.setEnabled(not active and self._scope_path is not None)
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
                self._shaft_controls(False)
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
