"""Measured overrides use the canonical target constructors through the worker."""

from typing import Any

from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from ..dialog_controls import save_cancel_buttons

MILLIMETRES_PER_METRE = 1000.0


class MeasuredTargetDialog(QDialog):
    def __init__(self, target: dict[str, Any], parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Measured Reference Dimensions")
        self._line = len(target["point_ids"]) == 2
        form = QFormLayout(self)
        explanation = QLabel(
            "Measure the actual reference. Use the distance between the marked endpoints, not the physical ends of a ruler. A new named reference preserves earlier observations."
        )
        explanation.setWordWrap(True)
        form.addRow(explanation)
        self.name = QLineEdit(
            "My Measured Ruler" if self._line else "My Measured Paper"
        )
        self.name.setMaxLength(150)
        form.addRow("Reference Name", self.name)
        self.length = self._dimension(target["object_points_m"][1][0])
        form.addRow("Marked Long Edge / Length", self.length)
        self.reference_width = self._dimension(
            0.001 if self._line else target["object_points_m"][3][2]
        )
        if not self._line:
            form.addRow("Width Across the Sheet", self.reference_width)
        buttons = save_cancel_buttons(self)
        form.addRow(buttons)

    def _dimension(self, metres: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.1, 100_000)
        spin.setDecimals(2)
        spin.setSuffix(" mm")
        spin.setValue(metres * MILLIMETRES_PER_METRE)
        return spin

    def parameters(self) -> dict[str, Any]:
        return {
            "reference_id": f"measured-{self.name.text().strip()}",
            "shape": "line" if self._line else "rectangle",
            "length_m": self.length.value() / MILLIMETRES_PER_METRE,
            "width_m": self.reference_width.value() / MILLIMETRES_PER_METRE,
        }
