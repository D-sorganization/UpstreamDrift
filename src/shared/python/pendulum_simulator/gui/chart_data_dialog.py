"""
Data selection dialog for pop-out charts.

Lets the user pick any two simulation variables for X and Y axes,
with optional regression fitting.

Design by Contract
------------------
- get_selection() returns (x_key, y_key, regression_degree) or None if cancelled.
- All available series are presented in a sorted dropdown.
"""

from __future__ import annotations

import logging

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..data_extractor import list_available_series

logger = logging.getLogger(__name__)


class ChartDataDialog(QDialog):
    """Dialog for selecting X/Y data series and regression options.

    Usage::

        dlg = ChartDataDialog(parent, model_type="double")
        if dlg.exec():
            x_key, y_key, reg_degree = dlg.get_selection()
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        model_type: str = "double",
    ) -> None:
        assert model_type is not None, "model_type must be provided"
        super().__init__(parent)
        self.setWindowTitle("Select Chart Data")
        self.setMinimumWidth(400)
        self._model_type = model_type
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Data selection group ---
        data_group = QGroupBox("Data Selection")
        form = QFormLayout()

        series = list_available_series(self._model_type)
        self._series_map: dict[str, tuple[str, str]] = {}

        self._x_combo = QComboBox()
        self._y_combo = QComboBox()

        for key, desc, unit in series:
            label = f"{desc} ({unit})"
            self._x_combo.addItem(label, key)
            self._y_combo.addItem(label, key)
            self._series_map[key] = (desc, unit)

        # Default: X=time, Y=torque_shoulder
        x_idx = self._x_combo.findData("time")
        if x_idx >= 0:
            self._x_combo.setCurrentIndex(x_idx)
        y_idx = self._y_combo.findData("tip_speed")
        if y_idx >= 0:
            self._y_combo.setCurrentIndex(y_idx)

        form.addRow("X axis:", self._x_combo)
        form.addRow("Y axis:", self._y_combo)
        data_group.setLayout(form)
        layout.addWidget(data_group)

        # --- Regression group ---
        reg_group = QGroupBox("Regression Fit")
        reg_form = QFormLayout()

        self._reg_degree = QSpinBox()
        self._reg_degree.setRange(0, 10)
        self._reg_degree.setValue(3)
        self._reg_degree.setToolTip("Polynomial degree (0 = no regression)")

        reg_form.addRow("Polynomial degree:", self._reg_degree)
        reg_group.setLayout(reg_form)
        layout.addWidget(reg_group)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selection(self) -> tuple[str, str, int]:
        """Return the user's selection.

        Returns
        -------
        (x_key, y_key, regression_degree)
        """
        x_key = self._x_combo.currentData()
        y_key = self._y_combo.currentData()
        degree = self._reg_degree.value()
        assert x_key is not None and y_key is not None
        return x_key, y_key, degree
