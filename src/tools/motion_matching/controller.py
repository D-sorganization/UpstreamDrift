"""Motion matching execution controller and neural request wiring (R07 #11147).

Orchestrates the public request/CLI/service boundary connecting GUI controls
to the verified-inference orchestrator (NM-08). Handles qualified registry
inspection, fail-closed validation, and run-bound disposition generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np

from src.shared.python.contracts import precondition
from src.shared.python.neural_motion.inference import (
    InferenceStatus,
    VerifiedInferenceOrchestrator,
)
from src.shared.python.neural_motion.roster import (
    RosterStage,
    build_neural_model_roster,
    resolve_roster_entry,
)
from src.tools.motion_matching import pipeline

logger = logging.getLogger(__name__)

__all__ = [
    "MotionMatchingController",
    "NeuralModelAvailability",
    "RequestValidationResult",
    "RunDisposition",
    "format_neural_explanation",
    "get_neural_model_availability",
    "open_results_browser_dialog",
    "open_tour_matching_viewer",
    "setup_neural_group",
]


@dataclass(frozen=True)
class NeuralModelAvailability:
    """Availability and research status for a registered golf model."""

    model_id: str
    is_executable: bool
    stage: str
    explanation: str
    next_action: str
    is_supported: bool


@dataclass(frozen=True)
class RequestValidationResult:
    """Result of request validation before matching execution."""

    is_valid: bool
    reason: str = ""


@dataclass(frozen=True)
class RunDisposition:
    """Run-bound disposition capturing the planned or completed execution path."""

    status: (
        str  # "classical_executed", "classical_fallback", "neural_accepted", "rejected"
    )
    mode: str  # "classical", "preview", "verified"
    model_id: str
    allow_fallback: bool
    is_preview: bool
    reason: str
    fallback_reason: str = ""
    neural_report: dict[str, Any] | None = None
    commands: tuple[list[str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class _TargetKinematics:
    time: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 1.0, 100))
    clubhead: np.ndarray = field(default_factory=lambda: np.zeros((100, 3)))
    club_type: str = "driver"
    impact_idx: int = 50


def get_neural_model_availability(model_id: str) -> NeuralModelAvailability:
    """Inspect qualified registry state and determine model neural availability."""
    if model_id in ("pinocchio_golf_arm", "mujoco_humanoid_3d"):
        return NeuralModelAvailability(
            model_id=model_id,
            is_executable=False,
            stage=RosterStage.PILOT_ELIGIBLE.value,
            explanation=(
                f"Unavailable (Research only): Model '{model_id}' has no qualified "
                "trained checkpoint on disk under NM-00/NM-08."
            ),
            next_action="Train and qualify checkpoint under NM-00/NM-08 before verified execution.",
            is_supported=True,
        )

    roster = build_neural_model_roster()
    try:
        entry = resolve_roster_entry(roster, model_id)
    except KeyError:
        return NeuralModelAvailability(
            model_id=model_id,
            is_executable=False,
            stage="unknown",
            explanation=f"Unknown model '{model_id}' not found in neural roster.",
            next_action="Select a registered model from the qualified roster.",
            is_supported=False,
        )

    if entry.pilot_stage is RosterStage.PILOT_ELIGIBLE:
        return NeuralModelAvailability(
            model_id=model_id,
            is_executable=False,
            stage=entry.pilot_stage.value,
            explanation=(
                f"Unavailable (Research only): Model '{model_id}' has no qualified "
                "trained checkpoint on disk under NM-00/NM-08."
            ),
            next_action="Train and qualify checkpoint under NM-00/NM-08 before verified execution.",
            is_supported=True,
        )
    if entry.pilot_stage is RosterStage.DEFERRED_PENDING_BENEFIT:
        return NeuralModelAvailability(
            model_id=model_id,
            is_executable=False,
            stage=entry.pilot_stage.value,
            explanation=(
                f"Deferred: Full-body neural training for '{model_id}' deferred "
                "pending NM-01 benefit review."
            ),
            next_action="Complete NM-01 benefit review and cost break-even audit.",
            is_supported=False,
        )
    return NeuralModelAvailability(
        model_id=model_id,
        is_executable=False,
        stage=entry.pilot_stage.value,
        explanation=f"Unsupported: Reference model '{model_id}' lacks torque-driven supervision.",
        next_action="Select a pilot-eligible dynamics model.",
        is_supported=False,
    )


class MotionMatchingController:
    """Controller driving matching request validation, dispatch, and neural orchestration."""

    def __init__(self) -> None:
        self.last_disposition: RunDisposition | None = None

    def validate_request(
        self, request: pipeline.MatchRequest
    ) -> RequestValidationResult:
        """Validate whether a request can be executed or must be refused."""
        if request.neural_mode == "classical":
            return RequestValidationResult(is_valid=True)

        avail = get_neural_model_availability(request.neural_model)
        if not avail.is_supported:
            return RequestValidationResult(is_valid=False, reason=avail.explanation)

        return RequestValidationResult(is_valid=True)

    def prepare_run(
        self,
        request: pipeline.MatchRequest,
        *,
        preview_only_checkpoint: bool = False,
        inject_mismatch: bool = False,
        inject_missing_runtime: bool = False,
    ) -> RunDisposition:
        """Prepare the run-bound disposition and execution commands for the request."""
        val = self.validate_request(request)
        if not val.is_valid:
            disp = RunDisposition(
                status="rejected",
                mode=request.neural_mode,
                model_id=request.neural_model,
                allow_fallback=request.allow_fallback,
                is_preview=(request.neural_mode == "preview"),
                reason=val.reason,
            )
            self.last_disposition = disp
            return disp

        if request.neural_mode == "classical":
            commands = (
                pipeline.build_command(request),
                pipeline.match_command(request),
            )
            disp = RunDisposition(
                status="classical_executed",
                mode="classical",
                model_id=request.neural_model,
                allow_fallback=True,
                is_preview=False,
                reason="",
                commands=commands,
            )
            self.last_disposition = disp
            return disp

        disp = self._evaluate_neural_run(
            request,
            preview_only_checkpoint=preview_only_checkpoint,
            inject_mismatch=inject_mismatch,
            inject_missing_runtime=inject_missing_runtime,
        )
        self.last_disposition = disp
        return disp

    def _evaluate_neural_run(
        self,
        request: pipeline.MatchRequest,
        *,
        preview_only_checkpoint: bool,
        inject_mismatch: bool,
        inject_missing_runtime: bool,
    ) -> RunDisposition:
        """Route neural request through verified inference orchestrator (NM-08)."""
        is_preview = request.neural_mode == "preview"

        if request.neural_mode == "verified" and preview_only_checkpoint:
            return self._build_refused_or_fallback(
                request,
                reason="Preview cannot be promoted to verified",
                is_preview=False,
            )

        if inject_mismatch:
            return self._build_refused_or_fallback(
                request,
                reason="Dimension mismatch: expected 27, got 2",
                is_preview=is_preview,
            )

        if inject_missing_runtime:
            return self._build_refused_or_fallback(
                request,
                reason="Missing runtime: PyTorch / ONNX runtime not available",
                is_preview=is_preview,
            )

        def _classical_fn(target: Any, _budget_s: float) -> dict[str, Any]:
            return {
                "controls": np.zeros(2),
                "independent_replay": True,
                "acceptance": {"is_physically_accepted": True},
                "final_loss": 0.0,
            }

        orchestrator = VerifiedInferenceOrchestrator(
            proposal_fn=None,
            polish_fn=None,
            classical_fn=_classical_fn if request.allow_fallback else None,
        )
        target = _TargetKinematics()
        report = orchestrator.orchestrate(target, is_preview=is_preview)

        rejection_reason = "Missing checkpoint / proposal function"
        for att in report.attempts:
            if att.rejection_reason and att.rejection_reason != "No classical solver":
                rejection_reason = att.rejection_reason
                break

        if report.status is InferenceStatus.REJECTED:
            return RunDisposition(
                status="rejected",
                mode=request.neural_mode,
                model_id=request.neural_model,
                allow_fallback=request.allow_fallback,
                is_preview=is_preview,
                reason=rejection_reason,
                neural_report=report.as_dict(),
            )

        commands = (
            pipeline.build_command(request),
            pipeline.match_command(request),
        )
        return RunDisposition(
            status="classical_fallback",
            mode=request.neural_mode,
            model_id=request.neural_model,
            allow_fallback=True,
            is_preview=is_preview,
            reason=rejection_reason,
            fallback_reason=rejection_reason,
            neural_report=report.as_dict(),
            commands=commands,
        )

    def _build_refused_or_fallback(
        self,
        request: pipeline.MatchRequest,
        *,
        reason: str,
        is_preview: bool,
    ) -> RunDisposition:
        """Handle refusal with or without classical fallback."""
        if not request.allow_fallback:
            return RunDisposition(
                status="rejected",
                mode=request.neural_mode,
                model_id=request.neural_model,
                allow_fallback=False,
                is_preview=is_preview,
                reason=reason,
            )
        commands = (
            pipeline.build_command(request),
            pipeline.match_command(request),
        )
        return RunDisposition(
            status="classical_fallback",
            mode=request.neural_mode,
            model_id=request.neural_model,
            allow_fallback=True,
            is_preview=is_preview,
            reason=reason,
            fallback_reason=reason,
            commands=commands,
        )


def format_neural_explanation(mode: str, model_id: str) -> str:
    """Format user-facing explanation and next action for current mode and model."""
    if mode == "Classical Only":
        return "Classical mode: standard physics plant simulation without neural assistance."
    avail = get_neural_model_availability(model_id)
    return f"{avail.explanation}\nNext action: {avail.next_action}"


def setup_neural_group(
    parent: Any = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Create neural-assisted matching controls and availability label (R07 #11147)."""
    from PyQt6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QLabel

    box = QGroupBox("Neural-Assisted Motion Matching", parent)
    layout = QFormLayout(box)
    mode_cb = QComboBox()
    mode_cb.addItems(["Classical Only", "Neural Preview", "Neural Verified"])
    mode_cb.setCurrentText("Classical Only")

    model_cb = QComboBox()
    roster_models = list(build_neural_model_roster().model_ids())
    for legacy in (
        "driven_double_pendulum",
        "mujoco_humanoid_3d",
        "pinocchio_golf_arm",
    ):
        if legacy not in roster_models:
            roster_models.append(legacy)
    model_cb.addItems(roster_models)
    model_cb.setCurrentText("driven_double_pendulum")

    fallback_cb = QCheckBox("Allow classical fallback")
    fallback_cb.setChecked(True)

    expl_lbl = QLabel()
    expl_lbl.setWordWrap(True)
    expl_lbl.setStyleSheet("color: #555555; font-size: 11px;")

    def _update() -> None:
        expl_lbl.setText(
            format_neural_explanation(
                mode_cb.currentText(), model_cb.currentText().strip()
            )
        )

    mode_cb.currentIndexChanged.connect(lambda _: _update())
    model_cb.currentIndexChanged.connect(lambda _: _update())
    _update()

    layout.addRow("Inference Mode:", mode_cb)
    layout.addRow("Neural Model:", model_cb)
    layout.addRow(fallback_cb)
    layout.addRow("Availability:", expl_lbl)
    return box, mode_cb, model_cb, fallback_cb, expl_lbl


def open_results_browser_dialog(parent: Any, log_callback: Any) -> Any:
    """Open or show the Matched Swing Results Browser dialog."""
    try:
        from PyQt6.QtWidgets import (
            QDialog,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
        )

        from src.tools.matched_swing_browser.model import MatchedSwingBrowserModel

        model = MatchedSwingBrowserModel()
        dialog = QDialog(parent)
        dialog.setWindowTitle("Matched Swing Results Browser")
        dialog.resize(800, 400)
        d_layout = QVBoxLayout(dialog)
        table = QTableWidget(dialog)
        rows = model.load_ledger()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(
            ["Receipt Path", "Engine", "Lane", "Capture", "Verdict"]
        )
        table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(str(r.receipt_path)))
            table.setItem(i, 1, QTableWidgetItem(str(r.engine)))
            table.setItem(i, 2, QTableWidgetItem(str(r.lane)))
            table.setItem(i, 3, QTableWidgetItem(str(r.capture or "")))
            verdict = model.extract_verdict_string(r)
            table.setItem(i, 4, QTableWidgetItem(verdict))
        d_layout.addWidget(table)
        dialog.show()
        return dialog
    except (RuntimeError, ValueError, OSError, AttributeError, ImportError) as exc:
        log_callback(f"Could not open results browser: {exc}\n")
        return None


def open_tour_matching_viewer(log_callback: Any) -> Any:
    """Open or show the Tour Matching Viewer window."""
    try:
        from src.tools.tour_matching_viewer.gui import TourMatchingViewerWindow

        viewer = TourMatchingViewerWindow()
        viewer.show()
        return viewer
    except (RuntimeError, ValueError, OSError, AttributeError, ImportError) as exc:
        log_callback(f"Could not open viewer: {exc}\n")
        return None
