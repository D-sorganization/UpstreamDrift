"""PyQt6 Launch Monitor Analytics workbench."""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib.dates import date2num
from PyQt6 import QtCore, QtGui, QtWidgets

from src.tools.launch_monitor_model import (
    CorrelationResult,
    FilterRule,
    ImportedSession,
    ImportManifest,
    ImportOptions,
    LaunchMonitorProject,
    TreatmentConfig,
    analyze_dispersion,
    analyze_trend,
    apply_treatment,
    compare_monitors,
    compute_correlations,
    compute_pca,
    compute_vif,
    fit_predictive_model,
    import_session,
    load_private_corpus,
    numeric_metric_columns,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.launch_monitor_analytics.flexible_analysis_widget import (
    FlexibleAnalysisWidget,
)
from src.tools.launch_monitor_analytics.plot_canvas import PlotCanvas
from src.tools.launch_monitor_analytics.widgets import (
    DataFrameTable,
    ImportMappingDialog,
    populate_combo as _populate_combo,
    selected_text as _selected_text,
)

logger = get_logger(__name__)

__all__ = ["LaunchMonitorAnalyticsWindow", "MainWidget", "PlotCanvas", "main"]


class MainWidget(QtWidgets.QWidget):
    """Embeddable launch-monitor data-management and analysis workspace."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Launch Monitor Analytics")
        self.project = LaunchMonitorProject("Untitled Launch Monitor Study")
        self.analysis_frame = pd.DataFrame()
        self._dirty = False
        self._last_data_export: dict[str, str] | None = None
        self._build_ui()
        self._wire_actions()
        self._apply_theme_best_effort()
        self._refresh_all()

    def _build_ui(self) -> None:
        title = QtWidgets.QLabel("Launch Monitor Analytics")
        title.setObjectName("WorkspaceTitle")
        title_font = title.font()
        title_font.setPointSize(max(14, title_font.pointSize() + 4))
        title_font.setBold(True)
        title.setFont(title_font)
        subtitle = QtWidgets.QLabel(
            "Harmonize multi-vendor sessions, map interdependencies, compare "
            "measurement systems, and track dispersion and player change."
        )
        subtitle.setWordWrap(True)
        self.scientific_boundary = QtWidgets.QLabel(
            "Scientific Boundary: Correlation and predictive fit do not establish "
            "causality. Derived metrics and unmatched monitor comparisons require "
            "special care."
        )
        self.scientific_boundary.setObjectName("ScientificBoundary")
        self.scientific_boundary.setWordWrap(True)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_sessions_tab(), "Sessions")
        self.tabs.addTab(self._build_treatment_tab(), "Data Treatment")
        self.tabs.addTab(self._build_relationships_tab(), "Relationships")
        self.flexible_analysis = FlexibleAnalysisWidget()
        self.tabs.addTab(self.flexible_analysis, "Flexible Analysis")
        self.tabs.addTab(self._build_models_tab(), "Models")
        self.tabs.addTab(self._build_comparison_tab(), "Monitor Comparison")
        self.tabs.addTab(self._build_dispersion_tab(), "Dispersion")
        self.tabs.addTab(self._build_trends_tab(), "Trends")
        self.tabs.addTab(self._build_reports_tab(), "Reports")
        self.status_label = QtWidgets.QLabel(
            "Ready. Import one or more launch-monitor exports."
        )
        self.status_label.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.scientific_boundary)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status_label)

    def _build_sessions_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        self.session_tree = QtWidgets.QTreeWidget()
        self.session_tree.setHeaderLabels(["Session", "Monitor", "Shots", "Profile"])
        self.session_tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.data_table = DataFrameTable()
        self.import_button = QtWidgets.QPushButton("Import Files...")
        self.load_corpus_button = QtWidgets.QPushButton("Load Private Corpus")
        self.load_corpus_button.setToolTip(
            "Load every source in the authorized private shot corpus as one "
            "session per source. Requires LAUNCH_MONITOR_DATA_ROOT to point at "
            "a commit-pinned Launch-Monitor-Flight-Model-Campaign checkout."
        )
        self.open_project_button = QtWidgets.QPushButton("Open Project...")
        self.save_project_button = QtWidgets.QPushButton("Save Project...")
        self.remove_session_button = QtWidgets.QPushButton("Remove Selected Sessions")
        self.clear_project_button = QtWidgets.QPushButton("New Project")
        for button in (
            self.import_button,
            self.load_corpus_button,
            self.open_project_button,
            self.save_project_button,
            self.remove_session_button,
            self.clear_project_button,
        ):
            button.setMinimumHeight(30)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.load_corpus_button)
        buttons.addWidget(self.open_project_button)
        buttons.addWidget(self.save_project_button)
        buttons.addWidget(self.remove_session_button)
        buttons.addWidget(self.clear_project_button)
        buttons.addStretch(1)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.session_tree)
        splitter.addWidget(self.data_table)
        splitter.setSizes([340, 900])
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addLayout(buttons)
        layout.addWidget(splitter, 1)
        return tab

    def _build_treatment_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        self.required_metrics_edit = QtWidgets.QLineEdit()
        self.required_metrics_edit.setPlaceholderText("club_speed, ball_speed")
        self.outlier_metrics_edit = QtWidgets.QLineEdit()
        self.outlier_metrics_edit.setPlaceholderText("club_speed, ball_speed")
        self.outlier_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.outlier_threshold_spin.setRange(1.0, 20.0)
        self.outlier_threshold_spin.setValue(4.5)
        self.outlier_threshold_spin.setSingleStep(0.25)
        self.exclude_flagged_check = QtWidgets.QCheckBox(
            "Exclude Flagged Rows from Analysis View"
        )
        self.exclude_flagged_check.setChecked(False)
        self.run_treatment_button = QtWidgets.QPushButton(
            "Apply Reproducible Treatment"
        )
        self.filter_table = QtWidgets.QTableWidget(0, 3)
        self.filter_table.setHorizontalHeaderLabels(["Column", "Operator", "Value"])
        filter_header = self.filter_table.horizontalHeader()
        assert filter_header is not None
        filter_header.setStretchLastSection(True)
        self.add_filter_button = QtWidgets.QPushButton("Add Filter")
        self.remove_filter_button = QtWidgets.QPushButton("Remove Selected Filters")
        filter_buttons = QtWidgets.QHBoxLayout()
        filter_buttons.addWidget(self.add_filter_button)
        filter_buttons.addWidget(self.remove_filter_button)
        filter_buttons.addStretch(1)
        form = QtWidgets.QFormLayout()
        form.addRow("Required Metrics:", self.required_metrics_edit)
        form.addRow("Robust-Outlier Metrics:", self.outlier_metrics_edit)
        form.addRow("Modified Z Threshold:", self.outlier_threshold_spin)
        form.addRow("", self.exclude_flagged_check)
        form.addRow("", self.run_treatment_button)
        controls = QtWidgets.QGroupBox("Treatment Recipe")
        controls.setLayout(form)
        self.flags_table = DataFrameTable()
        self.audit_text = QtWidgets.QPlainTextEdit()
        self.audit_text.setReadOnly(True)
        self.audit_text.setPlaceholderText("Every treatment action is recorded here.")
        output = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        output.addWidget(self.flags_table)
        output.addWidget(self.audit_text)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(controls)
        filter_box = QtWidgets.QGroupBox("Structured Subset Filters")
        filter_layout = QtWidgets.QVBoxLayout(filter_box)
        filter_layout.addLayout(filter_buttons)
        filter_layout.addWidget(self.filter_table)
        layout.addWidget(filter_box)
        layout.addWidget(output, 1)
        return tab

    def _build_relationships_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        self.relationship_method = QtWidgets.QComboBox()
        self.relationship_method.addItems(["pearson", "spearman", "kendall"])
        self.relationship_metrics = QtWidgets.QListWidget()
        self.relationship_metrics.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.relationship_controls = QtWidgets.QListWidget()
        self.relationship_controls.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.edge_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.edge_threshold_spin.setRange(0.0, 1.0)
        self.edge_threshold_spin.setValue(0.3)
        self.edge_threshold_spin.setSingleStep(0.05)
        self.run_relationship_button = QtWidgets.QPushButton("Map Interdependencies")
        self.run_multivariate_button = QtWidgets.QPushButton(
            "Run PCA and VIF Diagnostics"
        )
        control_form = QtWidgets.QFormLayout()
        control_form.addRow("Method:", self.relationship_method)
        control_form.addRow("Metrics:", self.relationship_metrics)
        control_form.addRow("Partial-Correlation Controls:", self.relationship_controls)
        control_form.addRow("Network Edge Threshold:", self.edge_threshold_spin)
        control_form.addRow("", self.run_relationship_button)
        control_form.addRow("", self.run_multivariate_button)
        control_box = QtWidgets.QGroupBox("Relationship Configuration")
        control_box.setLayout(control_form)
        self.relationship_table = DataFrameTable()
        self.relationship_plot = PlotCanvas()
        output = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        output.addWidget(self.relationship_plot)
        output.addWidget(self.relationship_table)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(control_box)
        splitter.addWidget(output)
        splitter.setSizes([330, 900])
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(splitter, 1)
        return tab

    def _build_models_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        self.model_target = QtWidgets.QComboBox()
        self.model_features = QtWidgets.QListWidget()
        self.model_features.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.model_type = QtWidgets.QComboBox()
        self.model_type.addItems(["linear", "ridge", "lasso", "elastic_net", "mlp"])
        self.model_group = QtWidgets.QComboBox()
        self.model_group.addItems(
            ["(random split)", "session_id", "monitor_vendor", "club"]
        )
        self.model_seed = QtWidgets.QSpinBox()
        self.model_seed.setRange(0, 2_147_483_647)
        self.model_seed.setValue(42)
        self.run_model_button = QtWidgets.QPushButton("Fit and Validate Model")
        form = QtWidgets.QFormLayout()
        form.addRow("Target:", self.model_target)
        form.addRow("Features:", self.model_features)
        form.addRow("Model:", self.model_type)
        form.addRow("Grouped Holdout:", self.model_group)
        form.addRow("Random Seed:", self.model_seed)
        form.addRow("", self.run_model_button)
        controls = QtWidgets.QGroupBox("Predictive Recipe")
        controls.setLayout(form)
        self.model_report = QtWidgets.QPlainTextEdit()
        self.model_report.setReadOnly(True)
        self.model_plot = PlotCanvas()
        output = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        output.addWidget(self.model_plot)
        output.addWidget(self.model_report)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(output)
        splitter.setSizes([330, 900])
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(splitter, 1)
        return tab

    def _build_comparison_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        self.comparison_metric = QtWidgets.QComboBox()
        self.comparison_match = QtWidgets.QComboBox()
        self.comparison_reference = QtWidgets.QComboBox()
        self.run_comparison_button = QtWidgets.QPushButton("Compare Monitor Behavior")
        self.comparison_warning = QtWidgets.QLabel(
            "Use a match identifier for the same shots whenever possible. "
            "Unmatched results are descriptive, not calibration evidence."
        )
        self.comparison_warning.setWordWrap(True)
        form = QtWidgets.QFormLayout()
        form.addRow("Metric:", self.comparison_metric)
        form.addRow("Matched-Shot Column:", self.comparison_match)
        form.addRow("Reference Monitor:", self.comparison_reference)
        form.addRow("", self.run_comparison_button)
        form.addRow("", self.comparison_warning)
        controls = QtWidgets.QGroupBox("Comparison Design")
        controls.setLayout(form)
        self.comparison_table = DataFrameTable()
        self.comparison_plot = PlotCanvas()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        output = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        output.addWidget(self.comparison_plot)
        output.addWidget(self.comparison_table)
        splitter.addWidget(output)
        splitter.setSizes([330, 900])
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(splitter, 1)
        return tab

    def _build_dispersion_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        self.dispersion_forward = QtWidgets.QComboBox()
        self.dispersion_lateral = QtWidgets.QComboBox()
        self.dispersion_group = QtWidgets.QComboBox()
        self.dispersion_group.addItems(
            ["(all shots)", "monitor_vendor", "session_id", "club"]
        )
        self.run_dispersion_button = QtWidgets.QPushButton("Analyze Dispersion")
        form = QtWidgets.QFormLayout()
        form.addRow("Forward Coordinate:", self.dispersion_forward)
        form.addRow("Lateral Coordinate:", self.dispersion_lateral)
        form.addRow("Group By:", self.dispersion_group)
        form.addRow("", self.run_dispersion_button)
        controls = QtWidgets.QGroupBox("Dispersion Definition")
        controls.setLayout(form)
        self.dispersion_table = DataFrameTable()
        self.dispersion_plot = PlotCanvas()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        output = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        output.addWidget(self.dispersion_plot)
        output.addWidget(self.dispersion_table)
        splitter.addWidget(output)
        splitter.setSizes([330, 900])
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(splitter, 1)
        return tab

    def _build_trends_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        self.trend_metric = QtWidgets.QComboBox()
        self.trend_time = QtWidgets.QComboBox()
        self.trend_window = QtWidgets.QSpinBox()
        self.trend_window.setRange(3, 500)
        self.trend_window.setValue(10)
        self.run_trend_button = QtWidgets.QPushButton("Analyze Longitudinal Change")
        form = QtWidgets.QFormLayout()
        form.addRow("Metric:", self.trend_metric)
        form.addRow("Time Column:", self.trend_time)
        form.addRow("Rolling Window:", self.trend_window)
        form.addRow("", self.run_trend_button)
        controls = QtWidgets.QGroupBox("Trend Configuration")
        controls.setLayout(form)
        self.trend_table = DataFrameTable()
        self.trend_plot = PlotCanvas()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        output = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        output.addWidget(self.trend_plot)
        output.addWidget(self.trend_table)
        splitter.addWidget(output)
        splitter.setSizes([330, 900])
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(splitter, 1)
        return tab

    def _build_reports_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        self.report_text = QtWidgets.QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.export_data_button = QtWidgets.QPushButton("Export Canonical Data...")
        self.export_manifest_button = QtWidgets.QPushButton(
            "Export Reproducibility Manifest..."
        )
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.export_data_button)
        buttons.addWidget(self.export_manifest_button)
        buttons.addStretch(1)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addLayout(buttons)
        layout.addWidget(self.report_text, 1)
        return tab

    def _wire_actions(self) -> None:
        self.import_button.clicked.connect(self._on_import_files)
        self.load_corpus_button.clicked.connect(self._on_load_private_corpus)
        self.open_project_button.clicked.connect(self._on_open_project)
        self.save_project_button.clicked.connect(self._on_save_project)
        self.remove_session_button.clicked.connect(self._remove_selected_sessions)
        self.clear_project_button.clicked.connect(self.clear_project)
        self.run_treatment_button.clicked.connect(self._run_treatment_ui)
        self.add_filter_button.clicked.connect(self._add_filter_row)
        self.remove_filter_button.clicked.connect(self._remove_filter_rows)
        self.run_relationship_button.clicked.connect(self._run_relationship_ui)
        self.run_multivariate_button.clicked.connect(self._run_multivariate_ui)
        self.run_model_button.clicked.connect(self._run_model_ui)
        self.run_comparison_button.clicked.connect(self._run_comparison_ui)
        self.run_dispersion_button.clicked.connect(self._run_dispersion_ui)
        self.run_trend_button.clicked.connect(self._run_trend_ui)
        self.export_data_button.clicked.connect(self._on_export_data)
        self.export_manifest_button.clicked.connect(self._on_export_manifest)

    def _apply_theme_best_effort(self) -> None:
        try:
            from src.shared.python.theme import apply_theme_to_window
        except ImportError:
            return
        if apply_theme_to_window is None:
            return
        try:
            cast(Any, apply_theme_to_window)(self)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("Theme application skipped: %s", exc)

    def is_dirty(self) -> bool:
        return self._dirty

    def import_file(self, path: str | Path, options: ImportOptions | None = None):  # noqa: ANN201
        """Import and append a session, then refresh every workspace."""
        session = import_session(path, options)
        self.project.add_session(session)
        self.analysis_frame = self.project.combined_shots()
        self._dirty = True
        self._refresh_all()
        self.status_label.setText(
            f"Imported {session.name}: {len(session.shots)} shots from "
            f"{session.manifest.vendor}. Project now contains "
            f"{len(self.analysis_frame)} shots."
        )
        return session

    def load_private_corpus_sessions(self, root: str | Path | None = None) -> int:
        """Load the private shot corpus as one session per corpus source.

        Returns the number of sessions added. Sources already present in the
        project are skipped rather than raising, so the action is repeatable.
        Raises ``FileNotFoundError`` when no authorized checkout or corpus
        manifest is available, ``ValueError`` when the corpus disagrees with
        the manifest that describes it (the ADR-0048 D30 gate), and
        ``ImportError`` when the Parquet reader is not installed; callers in
        the UI surface those as dialogs.
        """
        frame = load_private_corpus(root)
        existing = {session.session_id for session in self.project.sessions}
        imported_at = datetime.now(UTC).isoformat()
        added = 0
        for session_id, group in frame.groupby("session_id", sort=True, observed=True):
            identifier = str(session_id)
            if identifier in existing:
                continue
            shots = group.reset_index(drop=True)
            vendor = str(shots["monitor_vendor"].iloc[0])
            manifest = ImportManifest(
                source_path=f"corpus://{identifier}",
                file_sha256="private-authority-parquet",
                profile_id="private_corpus",
                vendor=vendor,
                imported_at=imported_at,
                row_count=len(shots),
                source_columns=tuple(str(column) for column in shots.columns),
                metric_sources={},
                source_units={},
                unit_evidence={},
            )
            self.project.add_session(
                ImportedSession(
                    session_id=identifier,
                    name=identifier,
                    shots=shots,
                    manifest=manifest,
                )
            )
            added += 1
        if added:
            self.analysis_frame = self.project.combined_shots()
            self._dirty = True
            self._refresh_all()
        return added

    def _on_load_private_corpus(self) -> None:
        previous = QtWidgets.QApplication.overrideCursor()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            added = self.load_private_corpus_sessions()
        except (FileNotFoundError, ImportError, ValueError) as error:
            logger.info("Private corpus load unavailable: %s", error)
            QtWidgets.QMessageBox.warning(
                self, "Private corpus unavailable", str(error)
            )
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            if previous is not None:
                QtWidgets.QApplication.setOverrideCursor(previous)
        if not added:
            self.status_label.setText(
                "Private corpus already loaded; no new sources were added."
            )
            return
        self.status_label.setText(
            f"Loaded the private corpus: {added} sources, "
            f"{len(self.analysis_frame)} shots now in the project."
        )

    def save_project(self, path: str | Path) -> Path:
        saved = self.project.save(path)
        self._dirty = False
        self.status_label.setText(f"Saved project to {saved}.")
        return saved

    def load_project(self, path: str | Path) -> None:
        self.project = LaunchMonitorProject.load(path)
        self.analysis_frame = self.project.combined_shots()
        self._dirty = False
        self._refresh_all()
        self.status_label.setText(
            f"Loaded {self.project.name}: {len(self.analysis_frame)} shots."
        )

    def clear_project(self) -> None:
        self.project = LaunchMonitorProject("Untitled Launch Monitor Study")
        self.analysis_frame = pd.DataFrame()
        self._dirty = False
        self._last_data_export = None
        self._refresh_all()
        self.status_label.setText("New empty project. Import launch-monitor exports.")

    def _refresh_all(self) -> None:
        self._refresh_session_tree()
        self.data_table.set_frame(self.analysis_frame)
        self.flexible_analysis.set_frame(self.analysis_frame)
        self._reset_analysis_canvases()
        metrics = numeric_metric_columns(self.analysis_frame)
        for widget in (
            self.relationship_metrics,
            self.relationship_controls,
            self.model_features,
        ):
            widget.clear()
            widget.addItems(metrics)
        for combo in (
            self.model_target,
            self.comparison_metric,
            self.dispersion_forward,
            self.dispersion_lateral,
            self.trend_metric,
        ):
            _populate_combo(combo, metrics)
        columns = [str(column) for column in self.analysis_frame.columns]
        _populate_combo(self.comparison_match, ["(unmatched)", *columns])
        monitors = (
            sorted(self.analysis_frame["monitor_vendor"].dropna().astype(str).unique())
            if "monitor_vendor" in self.analysis_frame
            else []
        )
        _populate_combo(self.comparison_reference, monitors)
        time_columns = [
            column
            for column in columns
            if "time" in column or "date" in column or column == "captured_at"
        ]
        _populate_combo(self.trend_time, time_columns)
        self._set_preferred_combo(self.dispersion_forward, "carry_distance")
        self._set_preferred_combo(self.dispersion_lateral, "lateral_carry")
        self._refresh_report()

    @staticmethod
    def _set_preferred_combo(combo: QtWidgets.QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _reset_analysis_canvases(self) -> None:
        """Clear every analysis canvas so stale charts never outlive their data.

        Called from ``_refresh_all``, the single choke point reached by
        ``clear_project``, ``import_file``, ``load_private_corpus_sessions``,
        ``load_project``, ``_remove_selected_sessions``, and
        ``_run_treatment_ui``. Any of those changes the underlying
        project/session set, so a chart built against the previous data
        must not remain visible until (or unless) the user reruns that
        specific analysis against the new data.
        """
        message = f"{self.project.name}: run an analysis to populate this chart."
        for canvas in (
            self.relationship_plot,
            self.model_plot,
            self.comparison_plot,
            self.dispersion_plot,
            self.trend_plot,
        ):
            canvas.empty(message)

    def _title_with_project(self, text: str) -> str:
        """Append the current project name so a chart's data identity is visible."""
        return f"{text} — {self.project.name}"

    def _refresh_session_tree(self) -> None:
        self.session_tree.clear()
        for session in self.project.sessions:
            item = QtWidgets.QTreeWidgetItem(
                [
                    session.name,
                    session.manifest.vendor,
                    str(len(session.shots)),
                    session.manifest.profile_id,
                ]
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, session.session_id)
            item.setToolTip(0, session.manifest.source_path)
            self.session_tree.addTopLevelItem(item)
        self.session_tree.resizeColumnToContents(0)

    def _refresh_report(self) -> None:
        source_fields = len(
            [
                column
                for column in self.analysis_frame
                if str(column).startswith("source::")
            ]
        )
        metrics = numeric_metric_columns(self.analysis_frame)
        warnings = [
            warning
            for session in self.project.sessions
            for warning in session.manifest.warnings
        ]
        self.report_text.setPlainText(
            "Launch Monitor Analytics Project\n"
            "================================\n"
            f"Project: {self.project.name}\n"
            f"Sessions: {len(self.project.sessions)}\n"
            f"Shots: {len(self.analysis_frame)}\n"
            f"Canonical Numeric Metrics: {len(metrics)}\n"
            f"Retained Source Fields: {source_fields}\n"
            f"Import Warnings: {len(warnings)}\n\n"
            f"Recorded Treatment Actions: {len(self.project.audit_log)}\n\n"
            "Scientific Interpretation\n"
            "-------------------------\n"
            "Relationships describe association, not causation. Identity-derived "
            "metrics are marked by the metric registry. Matched shots are required "
            "for monitor bias and agreement claims; unmatched comparisons remain "
            "descriptive. Original source columns and per-file SHA-256 provenance "
            "are retained in the project."
        )

    def run_relationship_analysis(self) -> CorrelationResult:
        metrics = _selected_text(self.relationship_metrics)
        if len(metrics) < 2:
            raise ValueError("Select at least two relationship metrics")
        controls = tuple(
            item
            for item in _selected_text(self.relationship_controls)
            if item not in metrics
        )
        result = compute_correlations(
            self.analysis_frame,
            metrics=metrics,
            method=self.relationship_method.currentText(),
            controls=controls,
            edge_threshold=float(self.edge_threshold_spin.value()),
        )
        self.relationship_table.set_frame(
            result.coefficients.reset_index(names="metric")
        )
        axes = self.relationship_plot.reset_axes()
        image = axes.imshow(
            result.coefficients.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm"
        )
        axes.set_xticks(range(len(metrics)), metrics, rotation=45, ha="right")
        axes.set_yticks(range(len(metrics)), metrics)
        axes.set_title(self._title_with_project(f"{result.method.title()} Correlation"))
        self.relationship_plot.figure.colorbar(image, ax=axes, fraction=0.046)
        self.relationship_plot.draw_idle()
        self.status_label.setText(
            f"Mapped {len(result.edges)} screened dependency edges across "
            f"{len(metrics)} metrics. Interpret as association, not causation."
        )
        return result

    def _run_treatment_ui(self) -> None:
        try:
            required = tuple(
                item.strip()
                for item in self.required_metrics_edit.text().split(",")
                if item.strip()
            )
            outliers = tuple(
                item.strip()
                for item in self.outlier_metrics_edit.text().split(",")
                if item.strip()
            )
            result = apply_treatment(
                self.project.combined_shots(),
                TreatmentConfig(
                    required_metrics=required,
                    outlier_metrics=outliers,
                    robust_z_threshold=float(self.outlier_threshold_spin.value()),
                    exclude_flagged=self.exclude_flagged_check.isChecked(),
                    filters=self._filter_rules(),
                ),
            )
        except ValueError as exc:
            self._show_error("Data Treatment Failed", exc)
            return
        self.analysis_frame = result.data
        self.project.record_actions(result.audit_log)
        self.flags_table.set_frame(result.flags)
        self.audit_text.setPlainText(
            json.dumps(self.project.audit_log, indent=2, default=str)
        )
        self._dirty = True
        self._refresh_all()
        self.status_label.setText(
            f"Treatment complete: {len(result.flags)} flags; "
            f"{len(result.data)} shots in the analysis view. Raw sessions are unchanged."
        )

    def _run_relationship_ui(self) -> None:
        try:
            self.run_relationship_analysis()
        except ValueError as exc:
            self._show_error("Relationship Analysis Failed", exc)

    def _run_multivariate_ui(self) -> None:
        metrics = _selected_text(self.relationship_metrics)
        try:
            pca = compute_pca(self.analysis_frame, metrics=metrics)
            vif = compute_vif(self.analysis_frame, metrics=metrics)
        except ValueError as exc:
            self._show_error("Multivariate Analysis Failed", exc)
            return
        table = pca.loadings.copy()
        table.insert(0, "VIF", vif.values)
        self.relationship_table.set_frame(table.reset_index(names="metric"))
        axes = self.relationship_plot.reset_axes()
        axes.bar(
            pca.explained_variance_ratio.index,
            pca.explained_variance_ratio.to_numpy(dtype=float),
        )
        axes.set_ylim(0, 1)
        axes.set_ylabel("Explained Variance Ratio")
        axes.set_title(self._title_with_project("Principal-Component Variance"))
        axes.grid(True, axis="y", alpha=0.25)
        self.relationship_plot.draw_idle()
        warning = ", ".join(vif.warning_metrics) or "none"
        self.status_label.setText(
            f"PCA/VIF complete for {pca.sample_count} complete shots. "
            f"VIF >= 5: {warning}."
        )

    def _run_model_ui(self) -> None:
        group = self.model_group.currentText()
        try:
            result = fit_predictive_model(
                self.analysis_frame,
                target=self.model_target.currentText(),
                features=_selected_text(self.model_features),
                model=self.model_type.currentText(),
                random_seed=int(self.model_seed.value()),
                group_column=None if group == "(random split)" else group,
            )
        except (ValueError, ImportError) as exc:
            self._show_error("Predictive Model Failed", exc)
            return
        self.model_report.setPlainText(
            json.dumps(
                {
                    "model": result.model,
                    "target": result.target,
                    "features": result.features,
                    "metrics": result.metrics,
                    "coefficients": result.coefficients,
                    "random_seed": result.random_seed,
                    "train_count": result.train_count,
                    "test_count": result.test_count,
                },
                indent=2,
            )
        )
        axes = self.model_plot.reset_axes()
        axes.scatter(
            result.predictions["actual"], result.predictions["predicted"], alpha=0.75
        )
        bounds = [
            result.predictions[["actual", "predicted"]].min().min(),
            result.predictions[["actual", "predicted"]].max().max(),
        ]
        axes.plot(bounds, bounds, linestyle="--", color="gray")
        axes.set_xlabel("Actual")
        axes.set_ylabel("Predicted")
        axes.set_title(
            self._title_with_project(
                f"Held-Out {result.model.replace('_', ' ').title()} Predictions"
            )
        )
        axes.grid(True, alpha=0.25)
        self.model_plot.draw_idle()
        self.status_label.setText(
            f"Model complete: R2={result.metrics['r2']:.3f}, "
            f"RMSE={result.metrics['rmse']:.3g}."
        )

    def _run_comparison_ui(self) -> None:
        match = self.comparison_match.currentText()
        try:
            result = compare_monitors(
                self.analysis_frame,
                metric=self.comparison_metric.currentText(),
                match_column=None if match == "(unmatched)" else match,
                reference_monitor=self.comparison_reference.currentText() or None,
            )
        except ValueError as exc:
            self._show_error("Monitor Comparison Failed", exc)
            return
        self.comparison_table.set_frame(
            pd.DataFrame(asdict(item) for item in result.pairwise)
        )
        axes = self.comparison_plot.reset_axes()
        labels = [item.monitor for item in result.summaries]
        means = [item.mean for item in result.summaries]
        errors = [item.standard_deviation for item in result.summaries]
        axes.errorbar(labels, means, yerr=errors, fmt="o", capsize=5)
        axes.set_title(
            self._title_with_project(f"{result.metric}: Mean and Standard Deviation")
        )
        axes.grid(True, axis="y", alpha=0.25)
        self.comparison_plot.draw_idle()
        warning = next((item.warning for item in result.pairwise if item.warning), None)
        self.status_label.setText(
            warning or "Matched monitor agreement analysis complete."
        )

    def _run_dispersion_ui(self) -> None:
        forward = self.dispersion_forward.currentText()
        lateral = self.dispersion_lateral.currentText()
        group_column = self.dispersion_group.currentText()
        groups: list[tuple[object, pd.DataFrame]] = [("All Shots", self.analysis_frame)]
        if group_column != "(all shots)" and group_column in self.analysis_frame:
            groups = [
                (name, group)
                for name, group in self.analysis_frame.groupby(
                    group_column, dropna=False
                )
            ]
        rows: list[dict[str, object]] = []
        axes = self.dispersion_plot.reset_axes()
        try:
            for name, group in groups:
                result = analyze_dispersion(group, forward=forward, lateral=lateral)
                rows.append({"group": str(name), **asdict(result)})
                axes.scatter(
                    group[forward], group[lateral], alpha=0.55, label=str(name)
                )
                angle = np.linspace(0, 2 * np.pi, 200)
                ellipse = np.column_stack(
                    [
                        result.ellipse_major / 2 * np.cos(angle),
                        result.ellipse_minor / 2 * np.sin(angle),
                    ]
                )
                rotation = np.array(
                    [
                        [
                            np.cos(result.ellipse_angle_rad),
                            -np.sin(result.ellipse_angle_rad),
                        ],
                        [
                            np.sin(result.ellipse_angle_rad),
                            np.cos(result.ellipse_angle_rad),
                        ],
                    ]
                )
                rotated = ellipse @ rotation.T
                axes.plot(
                    rotated[:, 0] + result.mean_forward,
                    rotated[:, 1] + result.mean_lateral,
                )
        except ValueError as exc:
            self._show_error("Dispersion Analysis Failed", exc)
            return
        self.dispersion_table.set_frame(pd.DataFrame(rows))
        axes.set_xlabel(forward)
        axes.set_ylabel(lateral)
        axes.set_title(self._title_with_project("95% Dispersion Ellipses"))
        axes.grid(True, alpha=0.25)
        if len(groups) > 1:
            axes.legend(fontsize="small")
        self.dispersion_plot.draw_idle()
        self.status_label.setText(f"Dispersion complete for {len(groups)} group(s).")

    def _run_trend_ui(self) -> None:
        try:
            result = analyze_trend(
                self.analysis_frame,
                metric=self.trend_metric.currentText(),
                time_column=self.trend_time.currentText(),
                rolling_window=int(self.trend_window.value()),
            )
        except ValueError as exc:
            self._show_error("Trend Analysis Failed", exc)
            return
        candidates = pd.DataFrame(asdict(item) for item in result.change_candidates)
        self.trend_table.set_frame(candidates)
        axes = self.trend_plot.reset_axes()
        time_column = self.trend_time.currentText()
        axes.scatter(
            result.rolling[time_column],
            result.rolling["value"],
            s=16,
            alpha=0.4,
            label="Shots",
        )
        axes.plot(
            result.rolling[time_column],
            result.rolling["rolling_mean"],
            label="Rolling Mean",
        )
        axes.plot(result.rolling[time_column], result.rolling["ewma"], label="EWMA")
        for candidate in result.change_candidates:
            axes.axvline(
                date2num(candidate.captured_at.to_pydatetime()),
                linestyle="--",
                alpha=0.35,
            )
        axes.set_title(
            self._title_with_project(f"{result.metric}: Longitudinal Change")
        )
        axes.legend(fontsize="small")
        axes.grid(True, alpha=0.25)
        self.trend_plot.draw_idle()
        self.status_label.setText(
            f"Trend slope={result.slope_per_day:.4g}/day; "
            f"{len(result.change_candidates)} candidate change point(s)."
        )

    def _on_import_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Import Launch Monitor Exports",
            "",
            "Data Files (*.csv *.tsv *.txt *.xlsx *.xls *.json);;All Files (*)",
        )
        for raw_path in paths:
            path = Path(raw_path)
            try:
                dialog = ImportMappingDialog(path, self)
                if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                    continue
                self.import_file(path, dialog.import_options())
            except (ValueError, OSError) as exc:
                self._show_error(f"Could Not Import {path.name}", exc)

    def _remove_selected_sessions(self) -> None:
        selected = self.session_tree.selectedItems()
        if not selected:
            return
        for item in selected:
            session_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            self.project.remove_session(str(session_id))
        self.analysis_frame = self.project.combined_shots()
        self._dirty = True
        self._refresh_all()

    def _on_open_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Launch Monitor Project",
            "",
            "Launch Monitor Projects (*.lmproject);;All Files (*)",
        )
        if path:
            try:
                self.load_project(path)
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self._show_error("Could Not Open Project", exc)

    def _on_save_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Launch Monitor Project",
            "study.lmproject",
            "Launch Monitor Projects (*.lmproject)",
        )
        if path:
            try:
                self.save_project(path)
            except OSError as exc:
                self._show_error("Could Not Save Project", exc)

    def export_data(self, path: str | Path, *, as_parquet: bool = False) -> Path:
        """Export the analysis view, stamping a shared export ID + hash.

        Mirrors the ``ImportManifest`` provenance pattern: a fresh export
        ID/timestamp is embedded in the artifact (a leading ``#`` comment
        row for CSV, Parquet schema metadata for Parquet) and recorded on
        ``self._last_data_export`` with the file's own SHA-256, so a later
        ``export_manifest`` call can correlate the two on disk.
        """
        destination = Path(path)
        export_id = uuid.uuid4().hex
        exported_at = datetime.now(UTC).isoformat()
        is_parquet = as_parquet or destination.suffix.lower() == ".parquet"
        if is_parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(self.analysis_frame, preserve_index=False)
            metadata = dict(table.schema.metadata or {})
            metadata[b"export_id"] = export_id.encode("utf-8")
            metadata[b"exported_at"] = exported_at.encode("utf-8")
            table = table.replace_schema_metadata(metadata)
            pq.write_table(table, destination)
        else:
            with destination.open("w", encoding="utf-8", newline="") as handle:
                handle.write(f"# export_id={export_id} exported_at={exported_at}\n")
                self.analysis_frame.to_csv(handle, index=False)
        data_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
        self._last_data_export = {
            "export_id": export_id,
            "exported_at": exported_at,
            "data_file": destination.name,
            "data_sha256": data_sha256,
        }
        return destination

    def _on_export_data(self) -> None:
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Canonical Analysis View",
            "launch_monitor_data.csv",
            "CSV (*.csv);;Parquet (*.parquet)",
        )
        if not path:
            return
        try:
            self.export_data(path, as_parquet="Parquet" in selected_filter)
        except (OSError, ValueError, ImportError) as exc:
            self._show_error("Could Not Export Data", exc)
            return
        export_id = (self._last_data_export or {}).get("export_id", "")
        self.status_label.setText(f"Exported analysis view to {path} (ID {export_id}).")

    def export_manifest(self, path: str | Path) -> Path:
        """Write the reproducibility manifest, correlated to the last export.

        Embeds the last ``export_data`` call's ID, filename, and SHA-256
        under ``data_export`` (``None`` if no export ran yet) so the pair
        stays verifiably linked even once separated on disk.
        """
        destination = Path(path)
        payload = {
            "project": self.project.name,
            "sessions": [asdict(session.manifest) for session in self.project.sessions],
            "treatment_audit_log": self.project.audit_log,
            "analysis_rows": len(self.analysis_frame),
            "canonical_metrics": numeric_metric_columns(self.analysis_frame),
            "scientific_boundary": self.scientific_boundary.text(),
            "data_export": self._last_data_export,
        }
        destination.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
        return destination

    def _on_export_manifest(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Reproducibility Manifest",
            "launch_monitor_manifest.json",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            self.export_manifest(path)
        except OSError as exc:
            self._show_error("Could Not Export Manifest", exc)

    def _show_error(self, title: str, exc: Exception) -> None:
        logger.warning("%s: %s", title, exc)
        self.status_label.setText(f"{title}: {exc}")
        QtWidgets.QMessageBox.warning(self, title, str(exc))

    def _add_filter_row(self) -> None:
        row = self.filter_table.rowCount()
        self.filter_table.insertRow(row)
        column_combo = QtWidgets.QComboBox()
        column_combo.addItems([str(column) for column in self.analysis_frame.columns])
        operator_combo = QtWidgets.QComboBox()
        operator_combo.addItems(["eq", "ne", "lt", "le", "gt", "ge", "contains", "in"])
        self.filter_table.setCellWidget(row, 0, column_combo)
        self.filter_table.setCellWidget(row, 1, operator_combo)
        self.filter_table.setItem(row, 2, QtWidgets.QTableWidgetItem(""))

    def _remove_filter_rows(self) -> None:
        rows = sorted(
            {index.row() for index in self.filter_table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.filter_table.removeRow(row)

    def _filter_rules(self) -> tuple[FilterRule, ...]:
        rules: list[FilterRule] = []
        for row in range(self.filter_table.rowCount()):
            column_widget = self.filter_table.cellWidget(row, 0)
            operator_widget = self.filter_table.cellWidget(row, 1)
            value_item = self.filter_table.item(row, 2)
            if not isinstance(column_widget, QtWidgets.QComboBox) or not isinstance(
                operator_widget, QtWidgets.QComboBox
            ):
                continue
            rules.append(
                FilterRule(
                    column_widget.currentText(),
                    operator_widget.currentText(),
                    value_item.text() if value_item is not None else "",
                )
            )
        return tuple(rules)


class LaunchMonitorAnalyticsWindow(QtWidgets.QMainWindow):
    """Standalone window wrapper for :class:`MainWidget`."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Launch Monitor Analytics")
        self.resize(1440, 900)
        self.main_widget = MainWidget(self)
        self.setCentralWidget(self.main_widget)
        self._build_menu()

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        assert menu_bar is not None
        file_menu = menu_bar.addMenu("&File")
        help_menu = menu_bar.addMenu("&Help")
        assert file_menu is not None
        assert help_menu is not None
        load_corpus_action = QtGui.QAction("Load &Private Corpus", self)
        load_corpus_action.setStatusTip(
            "Load every source in the authorized private shot corpus."
        )
        load_corpus_action.triggered.connect(
            self.main_widget._on_load_private_corpus  # noqa: SLF001 - same tool
        )
        file_menu.addAction(load_corpus_action)
        file_menu.addSeparator()
        quit_action = QtGui.QAction("&Quit", self)
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        about_action = QtGui.QAction("&About Launch Monitor Analytics", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About Launch Monitor Analytics",
            "<b>Launch Monitor Analytics</b><br><br>"
            "A vendor-neutral workbench for auditable session aggregation, "
            "impact-parameter interdependency mapping, predictive modeling, "
            "monitor agreement, dispersion, and longitudinal trend analysis.<br><br>"
            "Association and predictive performance do not establish causality.",
        )


def main(argv: list[str] | None = None) -> int:
    """Run the standalone workbench and return the Qt exit code."""
    arguments = argv if argv is not None else sys.argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(arguments)
    window = LaunchMonitorAnalyticsWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
