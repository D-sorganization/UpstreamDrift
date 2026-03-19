"""
Widget that displays the 3x3 mass matrix with real-time values,
color-coded to distinguish diagonal (self-coupling) from off-diagonal
(cross-coupling) terms.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter
from PyQt6.QtWidgets import QWidget

from ..simulation_triple import TripleSimulationResult
from .matrix_widget_base import MatrixWidgetBase


class TripleMatrixWidget(MatrixWidgetBase):
    """Displays 3x3 mass matrix and force decomposition in real time."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: TripleSimulationResult | None = None

    def set_simulation(self, result: TripleSimulationResult) -> None:
        """Pre: result is not None and has at least one time step."""
        super().set_simulation(result)

    def get_matrix_size(self) -> tuple[int, int]:
        """Return (rows, cols) for 3x3 matrix."""
        return (3, 3)

    def get_matrix_entries(self, mc: dict) -> list:
        """Return list of matrix cell entries for 3x3."""
        return [
            (0, 0, "M11", mc["M11"], True),
            (0, 1, "M12", mc["M12"], False),
            (0, 2, "M13", mc["M13"], False),
            (1, 0, "M21", mc["M21"], False),
            (1, 1, "M22", mc["M22"], True),
            (1, 2, "M23", mc["M23"], False),
            (2, 0, "M31", mc["M31"], False),
            (2, 1, "M32", mc["M32"], False),
            (2, 2, "M33", mc["M33"], True),
        ]

    def get_column_labels(self) -> list[str]:
        """Return DOF labels for 3-DOF system."""
        return ["shoulder", "elbow", "wrist"]

    def _draw_coupling_ratio(self, painter: QPainter, mc: dict, y: int) -> int:
        """Draw average coupling ratio for 3x3 matrix."""
        assert painter is not None, "painter must be provided"
        diag = np.array([mc["M11"], mc["M22"], mc["M33"]])
        off = np.array(
            [
                mc["M12"],
                mc["M13"],
                mc["M21"],
                mc["M23"],
                mc["M31"],
                mc["M32"],
            ]
        )
        denom = np.mean(np.abs(diag)) if np.any(diag) else 1.0
        ratio = min(np.mean(np.abs(off)) / max(denom, 1e-12), 1.0)

        bar_x = 20
        bar_w = self.width() - 40
        bar_h = 22

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(50, 50, 60)))
        painter.drawRoundedRect(QRectF(bar_x, y, bar_w, bar_h), 4, 4)

        painter.setBrush(QBrush(self.COLOR_OFFDIAG))
        painter.drawRoundedRect(QRectF(bar_x, y, bar_w * ratio, bar_h), 4, 4)

        painter.setPen(self.COLOR_TEXT)
        painter.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(bar_x, y, bar_w, bar_h),
            Qt.AlignmentFlag.AlignCenter,
            f"avg |Moff|/|Mdiag| = {ratio:.1%}",
        )

        return y + bar_h + 6
