"""Embeddable PyQt controls for the flexible analysis contract."""

from __future__ import annotations

import json

import pandas as pd
from PyQt6 import QtCore, QtWidgets

from src.shared.python.launch_monitor import (
    FlexibleAnalysisRequest,
    FlexibleAnalysisResult,
    analyze_variables,
)
from src.tools.launch_monitor_analytics.widgets import DataFrameTable


class FlexibleAnalysisWidget(QtWidgets.QWidget):
    """Select arbitrary variables and inspect traceable statistical results."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self.last_result: FlexibleAnalysisResult | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.outcome_combo = QtWidgets.QComboBox()
        self.predictor_list = QtWidgets.QListWidget()
        self.predictor_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["comprehensive", "correlation", "regression"])
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["pearson", "spearman", "kendall"])
        self.missing_combo = QtWidgets.QComboBox()
        self.missing_combo.addItems(["pairwise", "listwise", "fail"])
        self.group_combo = QtWidgets.QComboBox()
        self.min_samples_spin = QtWidgets.QSpinBox()
        self.min_samples_spin.setRange(3, 1_000_000)
        self.min_samples_spin.setValue(10)
        self.confidence_spin = QtWidgets.QDoubleSpinBox()
        self.confidence_spin.setRange(0.51, 0.999)
        self.confidence_spin.setDecimals(3)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setValue(0.95)
        self.run_button = QtWidgets.QPushButton("Run Flexible Analysis")
        self.run_button.clicked.connect(self.run_analysis)

        controls = (
            (self.outcome_combo, "Outcome Variable"),
            (self.predictor_list, "Predictor Variables"),
            (self.mode_combo, "Analysis Mode"),
            (self.method_combo, "Correlation Method"),
            (self.missing_combo, "Missing-Data Policy"),
            (self.group_combo, "Optional Grouping Variable"),
            (self.min_samples_spin, "Minimum Sample Count"),
            (self.confidence_spin, "Confidence Level"),
            (self.run_button, "Run Flexible Analysis"),
        )
        for control, accessible_name in controls:
            control.setAccessibleName(accessible_name)
            control.setToolTip(accessible_name)

        form = QtWidgets.QFormLayout()
        form.addRow("Outcome:", self.outcome_combo)
        form.addRow("Predictors:", self.predictor_list)
        form.addRow("Analysis Mode:", self.mode_combo)
        form.addRow("Correlation Method:", self.method_combo)
        form.addRow("Missing-Data Policy:", self.missing_combo)
        form.addRow("Group By:", self.group_combo)
        form.addRow("Minimum Samples:", self.min_samples_spin)
        form.addRow("Confidence Level:", self.confidence_spin)
        form.addRow(self.run_button)

        boundary = QtWidgets.QLabel(
            "Associations and fitted regressions do not establish causality. "
            "Aggregate observations are excluded from regression, and source-specific "
            "fields cannot be pooled across monitor vendors."
        )
        boundary.setWordWrap(True)
        boundary.setAccessibleName("Flexible Analysis Scientific Boundary")
        self.summary_table = DataFrameTable()
        self.summary_table.setAccessibleName("Flexible Analysis Results")
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setAccessibleName("Flexible Analysis Traceable Details")
        self.details.setPlaceholderText(
            "Run an analysis to inspect provenance and diagnostics."
        )

        output = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        output.addWidget(self.summary_table)
        output.addWidget(self.details)
        output.setSizes([300, 300])
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(boundary)
        left.addLayout(form)
        left.addStretch(1)
        layout.addLayout(left, 1)
        layout.addWidget(output, 2)

    def set_frame(self, frame: pd.DataFrame) -> None:
        """Replace analysis data and expose every usable numeric source field."""

        self._frame = frame
        numeric = [
            str(column)
            for column in frame.columns
            if pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 3
        ]
        previous_outcome = self.outcome_combo.currentText()
        self.outcome_combo.clear()
        self.outcome_combo.addItems(numeric)
        if previous_outcome in numeric:
            self.outcome_combo.setCurrentText(previous_outcome)
        self.predictor_list.clear()
        self.predictor_list.addItems(numeric)
        group_columns = [
            str(column)
            for column in frame.columns
            if frame[column].notna().any() and frame[column].nunique(dropna=True) <= 100
        ]
        self.group_combo.clear()
        self.group_combo.addItem("(none)")
        self.group_combo.addItems(group_columns)
        self.last_result = None
        self.summary_table.set_frame(pd.DataFrame())
        self.details.clear()

    def selected_predictors(self) -> tuple[str, ...]:
        return tuple(item.text() for item in self.predictor_list.selectedItems())

    def run_analysis(self) -> FlexibleAnalysisResult:
        """Execute the shared analysis contract and render its complete evidence."""

        group = self.group_combo.currentText()
        request = FlexibleAnalysisRequest(
            outcome=self.outcome_combo.currentText(),
            predictors=self.selected_predictors(),
            analysis_mode=self.mode_combo.currentText(),  # type: ignore[arg-type]
            correlation_method=self.method_combo.currentText(),  # type: ignore[arg-type]
            missing_policy=self.missing_combo.currentText(),  # type: ignore[arg-type]
            group_by=None if group == "(none)" else group,
            confidence_level=self.confidence_spin.value(),
            min_samples=self.min_samples_spin.value(),
        )
        result = analyze_variables(self._frame, request)
        rows = [
            {
                "predictor": item.predictor,
                "correlation": item.coefficient,
                "p_value": item.p_value,
                "adjusted_p_value": item.adjusted_p_value,
                "ci_lower": item.ci_lower,
                "ci_upper": item.ci_upper,
                "sample_count": item.sample_count,
            }
            for item in result.correlations
        ]
        regression = result.regression
        if regression:
            coefficients = regression.coefficients
            for name, item in coefficients.items():
                rows.append(
                    {
                        "predictor": f"OLS: {name}",
                        "coefficient": item.estimate,
                        "p_value": item.p_value,
                        "ci_lower": item.ci_lower,
                        "ci_upper": item.ci_upper,
                        "sample_count": regression.sample_count,
                    }
                )
        self.summary_table.set_frame(pd.DataFrame(rows))
        self.details.setPlainText(
            json.dumps(result.to_dict(), indent=2, sort_keys=True)
        )
        self.last_result = result
        return result


__all__ = ["FlexibleAnalysisWidget"]
