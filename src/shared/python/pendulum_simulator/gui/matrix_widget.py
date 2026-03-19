"""
Widget that displays the 2x2 mass matrix with real-time values,
color-coded to distinguish diagonal (self-coupling) from off-diagonal
(cross-coupling) terms.
"""

from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter
from PyQt6.QtWidgets import QWidget

from ..simulation import SimulationResult
from .matrix_widget_base import MatrixWidgetBase


class MatrixWidget(MatrixWidgetBase):
    """Displays 2x2 mass matrix and force decomposition in real time.

    The mass matrix is rendered as a 2x2 grid:
        - Diagonal terms (M11, M22) in blue — self-coupling
        - Off-diagonal terms (M12, M21) in orange — cross-coupling

    Below the matrix, the current torque balance is shown:
        M * qddot = tau - C(q,qdot)*qdot - G(q)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: SimulationResult | None = None

    def set_simulation(self, result: SimulationResult) -> None:
        """Pre: result is not None and has at least one time step."""
        super().set_simulation(result)

    def get_matrix_size(self) -> tuple[int, int]:
        """Return (rows, cols) for 2x2 matrix."""
        return (2, 2)

    def get_matrix_entries(self, mc: dict) -> list:
        """Return list of matrix cell entries for 2x2."""
        return [
            (0, 0, "M11", mc["M11"], True),
            (0, 1, "M12", mc["M12"], False),
            (1, 0, "M21", mc["M21"], False),
            (1, 1, "M22", mc["M22"], True),
        ]

    def get_column_labels(self) -> list[str]:
        """Return DOF labels for 2-DOF system."""
        return ["shoulder", "wrist"]

    def _draw_coupling_ratio(self, painter: QPainter, mc: dict, y: int) -> int:
        """Draw the ratio |M12/M11| as a bar and percentage."""
        assert painter is not None, "painter must be provided"
        ratio = abs(mc["M12"]) / mc["M11"] if mc["M11"] > 1e-12 else 0.0
        ratio = min(ratio, 1.0)

        bar_x = 20
        bar_w = self.width() - 40
        bar_h = 22

        # Background bar
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(50, 50, 60)))
        painter.drawRoundedRect(QRectF(bar_x, y, bar_w, bar_h), 4, 4)

        # Filled portion
        painter.setBrush(QBrush(self.COLOR_OFFDIAG))
        painter.drawRoundedRect(QRectF(bar_x, y, bar_w * ratio, bar_h), 4, 4)

        # Text
        painter.setPen(self.COLOR_TEXT)
        painter.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(bar_x, y, bar_w, bar_h),
            Qt.AlignmentFlag.AlignCenter,
            f"|M12/M11| = {ratio:.1%}",
        )

        return y + bar_h + 6
