"""Tab construction for the Launch Monitor Analytics workbench."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from PyQt6 import QtCore, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.async_action import AsyncActionBar
from src.tools.launch_monitor_analytics.flexible_analysis_widget import (
    FlexibleAnalysisWidget,
)
from src.tools.launch_monitor_analytics.plot_canvas import PlotCanvas
from src.tools.launch_monitor_analytics.reporting import (
    SCIENTIFIC_BOUNDARY_TEXT,
)
from src.tools.launch_monitor_analytics.widgets import (
    DataFrameTable,
)

logger = get_logger(__name__)


class _MainWidgetTabsMixin:
    """Widget construction, signal wiring and theming for ``MainWidget``."""

    if TYPE_CHECKING:
        # Host attributes/methods live on ``MainWidget`` (composed mixins).
        def __getattr__(self, name: str) -> Any: ...

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
        self.scientific_boundary = QtWidgets.QLabel(SCIENTIFIC_BOUNDARY_TEXT)
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
        # One shared bar for all seven analysis buttons: it disables every
        # trigger for the duration of any run, so a second click cannot queue
        # a second analysis on top of the first (#9470).
        self.action_bar = AsyncActionBar()
        self.action_bar.set_trigger_buttons(
            self.run_treatment_button,
            self.run_relationship_button,
            self.run_multivariate_button,
            self.run_model_button,
            self.run_comparison_button,
            self.run_dispersion_button,
            self.run_trend_button,
        )
        self.status_label = QtWidgets.QLabel(
            "Ready. Import one or more launch-monitor exports."
        )
        self.status_label.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.scientific_boundary)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.action_bar)
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
        self.remove_session_button = QtWidgets.QPushButton(
            "Remove Selected Sessions..."
        )
        self.remove_session_button.setToolTip("Confirmed; cannot be undone.")
        self.clear_project_button = QtWidgets.QPushButton("New Project...")
        self.clear_project_button.setToolTip("Offers to save unsaved work first.")
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
        self.remove_session_button.clicked.connect(self._on_remove_selected_sessions)
        self.clear_project_button.clicked.connect(self._on_new_project)
        self.run_treatment_button.clicked.connect(self._run_treatment_async)
        self.add_filter_button.clicked.connect(self._add_filter_row)
        self.remove_filter_button.clicked.connect(self._remove_filter_rows)
        self.run_relationship_button.clicked.connect(self._run_relationship_async)
        self.run_multivariate_button.clicked.connect(self._run_multivariate_async)
        self.run_model_button.clicked.connect(self._run_model_async)
        self.run_comparison_button.clicked.connect(self._run_comparison_async)
        self.run_dispersion_button.clicked.connect(self._run_dispersion_async)
        self.run_trend_button.clicked.connect(self._run_trend_async)
        # Analysis buttons run their compute off the GUI thread on the shared
        # action bar (#9470); the _run_*_ui methods stay as the synchronous
        # present(compute()) path.
        self.export_data_button.clicked.connect(self._on_export_data)
        self.export_manifest_button.clicked.connect(self._on_export_manifest)

    def _apply_theme_best_effort(self) -> None:
        try:
            from src.shared.python.theme import (  # type: ignore[attr-defined]
                apply_theme_to_window,
            )
        except ImportError:
            return
        if apply_theme_to_window is None:
            return
        try:
            cast(Any, apply_theme_to_window)(self)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("Theme application skipped: %s", exc)
