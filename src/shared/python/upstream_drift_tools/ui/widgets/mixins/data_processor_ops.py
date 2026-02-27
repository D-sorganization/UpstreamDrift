"""Data processor operations mixin.

Standardized data operations for DataProcessorWidget across the fleet.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from PyQt6.QtWidgets import QMessageBox, QWidget

if TYPE_CHECKING:
    from PyQt6.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QLineEdit,
        QSpinBox,
        QTextEdit,
    )

logger = logging.getLogger(__name__)


class DataProcessorOpsMixin:
    """Mixin supplying data-manipulation operations for DataProcessorWidget."""

    # -- Attributes provided by the composed class -----------------------
    filter_column: QComboBox
    filter_operator: QComboBox
    filter_value: QLineEdit
    query_input: QLineEdit
    agg_group_by: QComboBox
    agg_column: QComboBox
    agg_type: QComboBox
    new_col_name: QLineEdit
    new_col_expr: QLineEdit
    transform_column: QComboBox
    transform_type: QComboBox
    transform_param: QDoubleSpinBox
    rename_column: QComboBox
    rename_to: QLineEdit
    fit_x_column: QComboBox
    fit_y_column: QComboBox
    fit_type: QComboBox
    fit_degree: QSpinBox
    fit_results_text: QTextEdit
    engine: Any
    data_modified: Any  # pyqtSignal

    if TYPE_CHECKING:

        def _update_table(self) -> None: ...
        def _update_column_selectors(self) -> None: ...
        def refresh_statistics(self) -> None: ...
        def _set_status(self, message: str, success: bool = False) -> None: ...

    def _apply_filter(self) -> None:
        """Apply the quick filter."""
        column = self.filter_column.currentText()
        operator = self.filter_operator.currentText()
        value_str = self.filter_value.text()

        if not column or not value_str:
            QMessageBox.warning(
                cast(QWidget, self), "Filter Error", "Please specify column and value."
            )
            return

        try:
            value: Any = float(value_str)
        except ValueError:
            value = value_str

        result = self.engine.filter_data(column, operator, value)
        if result.success:
            self._update_table()
            self.refresh_statistics()
            self.data_modified.emit()
            self._set_status(result.message, success=True)
        else:
            QMessageBox.warning(cast(QWidget, self), "Filter Error", result.message)

    def _execute_query(self) -> None:
        """Execute a query expression."""
        expression = self.query_input.text()
        if not expression:
            QMessageBox.warning(
                cast(QWidget, self), "Query Error", "Please enter a query expression."
            )
            return

        result = self.engine.query(expression)
        if result.success:
            self._update_table()
            self.refresh_statistics()
            self.data_modified.emit()
            self._set_status(result.message, success=True)
        else:
            QMessageBox.warning(cast(QWidget, self), "Query Error", result.message)

    def _aggregate_data(self) -> None:
        """Perform data aggregation."""
        from ...data_processing.core import AggregationType

        group_by: str | None = self.agg_group_by.currentText()
        if group_by == "(None)":
            group_by = None

        column = self.agg_column.currentText()
        agg_type = AggregationType(self.agg_type.currentText())

        result = self.engine.aggregate(group_by, column, agg_type)
        if result.success:
            self._update_table()
            self._update_column_selectors()
            self.refresh_statistics()
            self.data_modified.emit()
            self._set_status(result.message, success=True)
        else:
            QMessageBox.warning(
                cast(QWidget, self), "Aggregation Error", result.message
            )

    def _add_column(self) -> None:
        """Add a calculated column."""
        name = self.new_col_name.text()
        expression = self.new_col_expr.text()

        if not name or not expression:
            QMessageBox.warning(
                cast(QWidget, self),
                "Error",
                "Please provide column name and expression.",
            )
            return

        result = self.engine.add_calculated_column(name, expression)
        if result.success:
            self._update_table()
            self._update_column_selectors()
            self.refresh_statistics()
            self.data_modified.emit()
            self._set_status(f"Added column: {name}", success=True)
        else:
            QMessageBox.warning(cast(QWidget, self), "Error", result.message)

    def _transform_column(self) -> None:
        """Apply a transformation to a column."""
        column = self.transform_column.currentText()
        transformation = self.transform_type.currentText()

        kwargs: dict[str, Any] = {}
        if transformation in ["round"]:
            kwargs["decimals"] = int(self.transform_param.value())
        elif transformation == "fillna":
            kwargs["value"] = self.transform_param.value()

        result = self.engine.transform_column(column, transformation, **kwargs)
        if result.success:
            self._update_table()
            self.refresh_statistics()
            self.data_modified.emit()
            self._set_status(result.message, success=True)
        else:
            QMessageBox.warning(cast(QWidget, self), "Transform Error", result.message)

    def _rename_column(self) -> None:
        """Rename a column."""
        old_name = self.rename_column.currentText()
        new_name = self.rename_to.text()

        if not new_name:
            QMessageBox.warning(
                cast(QWidget, self), "Error", "Please provide a new name."
            )
            return

        result = self.engine.rename_column(old_name, new_name)
        if result.success:
            self._update_table()
            self._update_column_selectors()
            self.data_modified.emit()
            self._set_status(result.message, success=True)
        else:
            QMessageBox.warning(cast(QWidget, self), "Error", result.message)

    def _drop_column(self) -> None:
        """Drop the selected column."""
        column = self.rename_column.currentText()
        reply = QMessageBox.question(
            cast(QWidget, self),
            "Drop Column",
            f"Are you sure you want to drop column '{column}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            result = self.engine.drop_columns([column])
            if result.success:
                self._update_table()
                self._update_column_selectors()
                self.refresh_statistics()
                self.data_modified.emit()
                self._set_status(result.message, success=True)

    def _fit_curve(self) -> None:
        """Perform curve fitting."""
        from ...data_processing.core import FitType

        x_col = self.fit_x_column.currentText()
        y_col = self.fit_y_column.currentText()
        fit_type = FitType(self.fit_type.currentText())
        degree = self.fit_degree.value()

        if not x_col or not y_col:
            QMessageBox.warning(
                cast(QWidget, self), "Error", "Please select X and Y columns."
            )
            return

        result = self.engine.fit_curve(x_col, y_col, fit_type, degree)
        if result:
            self.fit_results_text.setHtml(
                f"""
                <h4>Fit Results</h4>
                <p><b>Equation:</b> {result.equation}</p>
                <p><b>R-squared:</b> {result.r_squared:.6f}</p>
                <p><b>Coefficients:</b> {", ".join(f"{c:.6f}" for c in result.coefficients)}</p>
                <p><b>Residual Sum:</b> {sum(result.residuals**2):.6f}</p>
            """
            )
            self._set_status(f"Curve fit: R² = {result.r_squared:.4f}", success=True)
        else:
            self.fit_results_text.setText("Curve fitting failed. Check your data.")
