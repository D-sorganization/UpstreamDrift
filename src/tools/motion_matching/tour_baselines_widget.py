"""Tour Baselines widget for Motion Matching GUI (TB-11, #10596).

Provides the Qt widget tab for discovering, inspecting, comparing, and
reproducing tour baseline models with accessible badges, provenance, and
visual distinction semantics.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.tools.motion_matching import pipeline
from src.tools.motion_matching.tour_baselines_presenter import TourBaselinesPresenter

logger = logging.getLogger(__name__)


class TourBaselinesWidget(QWidget):
    """Tab widget exposing Tour Baselines in Motion Matching."""

    def __init__(
        self,
        parent: QWidget | None = None,
        presenter: TourBaselinesPresenter | None = None,
    ) -> None:
        super().__init__(parent)
        self._tb_presenter = presenter or TourBaselinesPresenter()
        self._init_ui()

    def _init_ui(self) -> None:
        # 1. Capture & Model Selectors
        sel_form = QFormLayout()
        self.tb_capture = QComboBox()
        self.tb_capture.addItems(["Driver", "7-Iron"])
        self.tb_capture.currentTextChanged.connect(self._on_tb_capture_changed)

        self.tb_model = QComboBox()
        self.tb_model.currentTextChanged.connect(self._on_tb_model_changed)

        self.tb_fit_mode = QComboBox()
        self.tb_fit_mode.addItems(
            ["Kinematic Pose", "Prescribed Trajectory", "Torque Driven"]
        )

        sel_form.addRow("Tour Capture:", self.tb_capture)
        sel_form.addRow("Model Preset:", self.tb_model)
        sel_form.addRow("Fit Mode:", self.tb_fit_mode)

        # 2. Metadata Display Group
        meta_group = QGroupBox("Model Metadata & Scientific Qualification")
        meta_form = QFormLayout(meta_group)

        self.tb_badge = QLabel("-")
        self.tb_ownership = QLabel("-")
        self.tb_support = QLabel("-")
        self.tb_rmse = QLabel("-")
        self.tb_residual = QLabel("-")
        self.tb_assumptions = QLabel("-")
        self.tb_markers = QLabel("-")
        self.tb_phases = QLabel("-")
        self.tb_blocker_info = QLabel("-")
        self.tb_budget = QLabel("-")

        meta_form.addRow("Qualification Status:", self.tb_badge)
        meta_form.addRow("Ownership:", self.tb_ownership)
        meta_form.addRow("Supported in UpstreamDrift:", self.tb_support)
        meta_form.addRow("Original Frame RMSE:", self.tb_rmse)
        meta_form.addRow("Projection Residual:", self.tb_residual)
        meta_form.addRow("Model Assumptions:", self.tb_assumptions)
        meta_form.addRow("Marker Sets:", self.tb_markers)
        meta_form.addRow("Phase Coverage:", self.tb_phases)
        meta_form.addRow("Active Blocker:", self.tb_blocker_info)
        meta_form.addRow("Compute Budget:", self.tb_budget)

        # 3. Where This Came From Collapsible Panel
        self.tb_toggle_where_btn = QPushButton("Toggle 'Where This Came From' Panel")
        self.tb_toggle_where_btn.clicked.connect(self._toggle_where_panel)

        self.tb_where_group = QGroupBox("Where This Came From (Provenance & Lineage)")
        self.tb_where_group.setHidden(True)
        where_form = QFormLayout(self.tb_where_group)

        self.tb_raw_hash = QLabel("-")
        self.tb_freq = QLabel("-")
        self.tb_preproc = QLabel("-")
        self.tb_geom = QLabel("-")
        self.tb_fit_cfg = QLabel("-")
        self.tb_receipt = QLabel("-")
        self.tb_limitations = QLabel("-")
        self.tb_limitations.setWordWrap(True)

        where_form.addRow("Raw Capture Hash:", self.tb_raw_hash)
        where_form.addRow("Capture Frequency:", self.tb_freq)
        where_form.addRow("Preprocessing Pipeline:", self.tb_preproc)
        where_form.addRow("Anthropometric Geometry:", self.tb_geom)
        where_form.addRow("Optimizer & Fit Config:", self.tb_fit_cfg)
        where_form.addRow("Replay Receipt:", self.tb_receipt)
        where_form.addRow("Scientific Limitations:", self.tb_limitations)

        # 4. Actions Row
        actions_row = QHBoxLayout()
        self.tb_open_btn = QPushButton("Open in Viewer")
        self.tb_clone_btn = QPushButton("Clone for Experiment")
        self.tb_compare_btn = QPushButton("Compare Models")
        self.tb_evidence_btn = QPushButton("Inspect Evidence")
        self.tb_reproduce_btn = QPushButton("Reproduce CLI")

        self.tb_open_btn.clicked.connect(self._on_tb_open)
        self.tb_clone_btn.clicked.connect(self._on_tb_clone)
        self.tb_compare_btn.clicked.connect(self._on_tb_compare)
        self.tb_evidence_btn.clicked.connect(self._on_tb_inspect_evidence)
        self.tb_reproduce_btn.clicked.connect(self._on_tb_reproduce)

        actions_row.addWidget(self.tb_open_btn)
        actions_row.addWidget(self.tb_clone_btn)
        actions_row.addWidget(self.tb_compare_btn)
        actions_row.addWidget(self.tb_evidence_btn)
        actions_row.addWidget(self.tb_reproduce_btn)

        # 5. Output Log & Results
        self.tb_log = QPlainTextEdit()
        self.tb_log.setReadOnly(True)
        self.tb_results = QLabel("Ready")
        self.tb_results.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(sel_form)
        layout.addWidget(meta_group)
        layout.addWidget(self.tb_toggle_where_btn)
        layout.addWidget(self.tb_where_group)
        layout.addLayout(actions_row)
        layout.addWidget(self.tb_log, stretch=1)
        layout.addWidget(self.tb_results)

        # Populate initial models
        self._populate_tb_models()

    def _populate_tb_models(self) -> None:
        capture = self.tb_capture.currentText().strip().lower()
        cap_key = "iron" if "iron" in capture else "driver"
        items = self._tb_presenter.list_models(capture=cap_key)
        self.tb_model.blockSignals(True)
        self.tb_model.clear()
        for it in items:
            self.tb_model.addItem(f"{it.name} ({it.badge_symbol})", it.model_id)
        self.tb_model.blockSignals(False)
        self._update_tour_baseline_details()

    def _on_tb_capture_changed(self, _text: str) -> None:
        self._populate_tb_models()

    def _on_tb_model_changed(self, _text: str) -> None:
        self._update_tour_baseline_details()

    def _toggle_where_panel(self) -> None:
        self.tb_where_group.setHidden(not self.tb_where_group.isHidden())

    def _get_current_tb_model_id(self) -> str:
        idx = self.tb_model.currentIndex()
        if idx >= 0:
            data = self.tb_model.itemData(idx)
            if data:
                return str(data)
        return "driven_double_pendulum"

    def _get_current_tb_capture(self) -> str:
        text = self.tb_capture.currentText().strip().lower()
        return "iron" if "iron" in text else "driver"

    def _update_tour_baseline_details(self) -> None:
        model_id = self._get_current_tb_model_id()
        capture = self._get_current_tb_capture()
        try:
            detail = self._tb_presenter.get_model_detail(model_id, capture)
            budget = self._tb_presenter.get_compute_budget(model_id)
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            self.tb_results.setText(f"Error loading details: {exc}")
            return

        self.tb_badge.setText(f"{detail.badge_symbol} {detail.badge_text}")
        if "[PASS]" in detail.badge_symbol:
            style = "font-weight: bold; color: forestgreen; background-color: honeydew; border: 1px solid green; border-radius: 4px; padding: 2px 6px;"
        elif "[REJECTED]" in detail.badge_symbol:
            style = "font-weight: bold; color: firebrick; background-color: mistyrose; border: 1px solid red; border-radius: 4px; padding: 2px 6px;"
        elif "[BLOCKED]" in detail.badge_symbol:
            style = "font-weight: bold; color: darkorange; background-color: cornsilk; border: 1px solid orange; border-radius: 4px; padding: 2px 6px;"
        elif "[REF]" in detail.badge_symbol:
            style = "font-weight: bold; color: royalblue; background-color: aliceblue; border: 1px solid blue; border-radius: 4px; padding: 2px 6px;"
        else:
            style = "font-weight: bold; color: dimgray; background-color: whitesmoke; border: 1px solid gray; border-radius: 4px; padding: 2px 6px;"
        self.tb_badge.setStyleSheet(style)

        self.tb_ownership.setText(detail.ownership)
        self.tb_support.setText(
            "Yes" if detail.supported else "No (requires native environment)"
        )

        err_str = (
            f"{detail.original_frame_error_mm:.2f} mm"
            if detail.original_frame_error_mm is not None
            else "-"
        )
        res_str = (
            f"{detail.projection_residual_mm:.2f} mm"
            if detail.projection_residual_mm is not None
            else "-"
        )
        self.tb_rmse.setText(err_str)
        self.tb_residual.setText(res_str)

        self.tb_assumptions.setText("; ".join(detail.model_assumptions))
        self.tb_markers.setText(
            f"Observed: {len(detail.observed_markers)} markers | Fitted: {len(detail.fitted_markers)} markers"
        )

        phases_active = [k.capitalize() for k, v in detail.phase_coverage.items() if v]
        self.tb_phases.setText(", ".join(phases_active))

        if detail.blocker_reason:
            blocker_text = (
                f"{detail.blocker_reason} (Governed by {detail.governing_issue})"
            )
        else:
            blocker_text = "None (No active blockers)"
        self.tb_blocker_info.setText(blocker_text)

        self.tb_budget.setText(
            f"Wall clock limit: {budget.max_wall_clock_s}s | "
            f"Max evaluations: {budget.max_evaluations} | "
            f"Parameter dim: {budget.parameter_dimension}"
        )

        where = detail.where_this_came_from
        self.tb_raw_hash.setText(where.raw_capture_hash)
        self.tb_freq.setText(f"{where.capture_frequency_hz} Hz")
        self.tb_preproc.setText(where.preprocessing)
        self.tb_geom.setText(where.geometry_spec)
        self.tb_fit_cfg.setText(where.fit_config)
        self.tb_receipt.setText(where.replay_receipt or "None (No receipt on disk)")
        self.tb_limitations.setText(" • " + "\n • ".join(where.scientific_limitations))

        self.tb_open_btn.setEnabled(detail.can_open)
        self.tb_clone_btn.setEnabled(detail.can_clone)
        self.tb_compare_btn.setEnabled(detail.can_compare)
        self.tb_evidence_btn.setEnabled(True)
        self.tb_reproduce_btn.setEnabled(True)

    def _on_tb_open(self) -> None:
        model_id = self._get_current_tb_model_id()
        capture = self._get_current_tb_capture()
        res = self._tb_presenter.open_baseline(model_id, capture)
        pkg_status = "Yes" if res.package is not None else "No (using default profile)"
        msg = (
            f"[OPEN] Model '{model_id}' on capture '{capture}'\n"
            f"View Mode: {res.view_type.upper()}\n"
            f"Observed Club: {res.observed_club_visual}\n"
            f"Simulated Club: {res.simulated_club_visual}\n"
            f"Package loaded: {pkg_status}\n"
        )
        self.tb_log.appendPlainText(msg)
        self.tb_results.setText(f"Opened {model_id} in {res.view_type} mode.")

    def _on_tb_clone(self) -> None:
        model_id = self._get_current_tb_model_id()
        capture = self._get_current_tb_capture()
        session_dir = (
            pipeline.REPO_ROOT / "artifacts" / "experiments" / f"{model_id}_{capture}"
        )
        exp_name = f"exp_{model_id}_{capture}"
        try:
            cloned_path = self._tb_presenter.clone_for_experiment(
                model_id=model_id,
                capture=capture,
                session_dir=session_dir,
                experiment_name=exp_name,
            )
            self.tb_log.appendPlainText(
                f"[CLONE] Successfully cloned preset into: {cloned_path}\n"
            )
            self.tb_results.setText(f"Preset cloned to {cloned_path.name}")
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            self.tb_log.appendPlainText(f"[CLONE ERROR] {exc}\n")
            self.tb_results.setText(f"Clone error: {exc}")

    def _on_tb_compare(self) -> None:
        model_a_id = self._get_current_tb_model_id()
        capture = self._get_current_tb_capture()
        model_b_id = (
            "driven_triple_pendulum"
            if model_a_id == "driven_double_pendulum"
            else "driven_double_pendulum"
        )
        report = self._tb_presenter.compare_models(model_a_id, model_b_id, capture)
        msg = (
            f"[COMPARE] {model_a_id} vs {model_b_id} on {capture}\n"
            f"Verdict: {report.verdict}\n"
            f"Metric Deltas: {report.metric_deltas}\n"
            f"Topology Diff: {report.topology_comparison}\n"
        )
        self.tb_log.appendPlainText(msg)
        self.tb_results.setText(report.verdict)

    def _on_tb_inspect_evidence(self) -> None:
        model_id = self._get_current_tb_model_id()
        capture = self._get_current_tb_capture()
        evidence = self._tb_presenter.inspect_evidence(model_id, capture)
        msg = (
            f"[EVIDENCE] Model: {evidence.model_id} | Capture: {evidence.capture}\n"
            f"Git Commit: {evidence.git_commit} | Engine Version: {evidence.engine_version}\n"
            f"Status Bundle: {evidence.status_bundle}\n"
            f"Receipt Path: {evidence.receipt_path}\n"
            f"Metrics: {evidence.metrics}\n"
        )
        self.tb_log.appendPlainText(msg)
        self.tb_results.setText(f"Evidence inspected for {model_id}.")

    def _on_tb_reproduce(self) -> None:
        model_id = self._get_current_tb_model_id()
        capture = self._get_current_tb_capture()
        cmd = self._tb_presenter.reproduce(model_id, capture)
        msg = f"[REPRODUCE] Command:\n$ {cmd}\n"
        self.tb_log.appendPlainText(msg)
        self.tb_results.setText("Reproduction command copied.")


__all__ = ["TourBaselinesWidget"]
