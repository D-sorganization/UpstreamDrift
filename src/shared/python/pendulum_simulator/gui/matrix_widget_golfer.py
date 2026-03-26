"""
Widget for displaying the 8x8 mass matrix, force balance, energy,
and constraint violation for the golfer model.
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..simulation_golfer import GolferSimulationResult
from .matrix_widget_base import MatrixWidgetBase

logger = logging.getLogger(__name__)


class GolferMatrixWidget(MatrixWidgetBase):
    """Displays 8x8 mass matrix, energy, forces, and constraints in real time."""

    COLOR_CONSTRAINT_OK = QColor(80, 200, 120)
    COLOR_CONSTRAINT_BAD = QColor(230, 80, 80)

    DOF_LABELS = ["Hub", "RS", "RE", "RH", "LS", "LE", "LH", "Club"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(280, 400)
        self._result: GolferSimulationResult | None = None

    def set_simulation(self, result: GolferSimulationResult) -> None:
        """Pre: result is not None and has at least one time step."""
        super().set_simulation(result)

    def get_matrix_size(self) -> tuple[int, int]:
        """Return (rows, cols) for 8x8 matrix."""
        return (8, 8)

    def get_matrix_entries(self, mc: dict) -> list:
        """Return list of matrix cell entries for 8x8 (numpy array)."""
        if not (mc is not None):
            raise ValueError("mc must be provided")
        entries = []
        for row in range(8):
            for col in range(8):
                label = f"M{row + 1}{col + 1}"
                value = mc[row, col]
                is_diag = row == col
                entries.append((row, col, label, value, is_diag))
        return entries

    def get_column_labels(self) -> list[str]:
        """Return DOF labels for golfer model."""
        return self.DOF_LABELS

    def paintEvent(self, event: object) -> None:
        """Override to include constraint violation section."""
        if not (event is not None):
            raise ValueError("event must be provided")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.COLOR_BG)

        if self._result is None:
            painter.setPen(self.COLOR_LABEL)
            painter.setFont(QFont("Sans", 11))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No simulation loaded",
            )
            painter.end()
            return

        y = 10
        y = self._draw_section_title(painter, "Mass Matrix M(q)  [8x8]", y)
        y = self._draw_mass_matrix_compact(painter, y)
        y += 8

        y = self._draw_section_title(painter, "Constraint Violation", y)
        y = self._draw_constraint_violation(painter, y)
        y += 8

        y = self._draw_section_title(painter, "Force Balance", y)
        y = self._draw_force_balance(painter, y)
        y += 8

        y = self._draw_section_title(painter, "Energy", y)
        y = self._draw_energy(painter, y)

        painter.end()

    def _draw_mass_matrix_compact(self, painter: QPainter, y: int) -> int:
        """Draw the 8x8 mass matrix in compact heat-map style."""
        if not (self._result is not None):
            raise ValueError("DbC Blocked: Precondition failed.")
        M = self._result.mass_matrix_at(self._current_idx)

        n = 8
        avail_w = self.width() - 60
        cell = max(16, min(32, avail_w // n))
        grid_w = n * cell
        margin_x = (self.width() - grid_w) // 2

        max_val = max(np.abs(M).max(), 1e-12)
        painter.setFont(QFont("Monospace", 7))

        for row in range(n):
            for col in range(n):
                cx = margin_x + col * cell
                cy = y + row * cell
                val = M[row, col]
                is_diag = row == col

                intensity = min(abs(val) / max_val, 1.0)
                if is_diag:
                    bg = QColor(
                        int(40 + 40 * intensity),
                        int(55 + 40 * intensity),
                        int(80 + 50 * intensity),
                    )
                    border = self.COLOR_DIAGONAL
                else:
                    bg = QColor(
                        int(50 + 60 * intensity),
                        int(40 + 40 * intensity),
                        int(30 + 20 * intensity),
                    )
                    border = self.COLOR_OFFDIAG

                rect = QRectF(cx, cy, cell, cell)
                painter.setPen(QPen(border, 1))
                painter.setBrush(QBrush(bg))
                painter.drawRect(rect)

                painter.setPen(self.COLOR_TEXT)
                if abs(val) > 0.01:
                    txt = f"{val:.1f}"
                else:
                    txt = f"{val:.0e}" if val != 0 else "0"
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)

        painter.setPen(self.COLOR_LABEL)
        painter.setFont(QFont("Sans", 6))
        for i, label in enumerate(self.DOF_LABELS):
            lx = margin_x + i * cell
            painter.drawText(lx, y - 2, label)
            ly = y + i * cell + cell // 2 + 3
            painter.drawText(margin_x - 28, ly, label)

        return y + n * cell + 8

    def _draw_constraint_violation(self, painter: QPainter, y: int) -> int:
        """Draw constraint violation as a progress bar."""
        if not (self._result is not None):
            raise ValueError("DbC Blocked: Precondition failed.")
        v = self._result.constraint_violation_at(self._current_idx)

        bar_x = 20
        bar_w = self.width() - 40
        bar_h = 20

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(50, 50, 60)))
        painter.drawRoundedRect(QRectF(bar_x, y, bar_w, bar_h), 4, 4)

        ratio = min(v * 1000, 1.0)
        color = self.COLOR_CONSTRAINT_OK if v < 1e-4 else self.COLOR_CONSTRAINT_BAD
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(QRectF(bar_x, y, bar_w * ratio, bar_h), 4, 4)

        painter.setPen(self.COLOR_TEXT)
        painter.setFont(QFont("Monospace", 9, QFont.Weight.Bold))
        painter.drawText(
            QRectF(bar_x, y, bar_w, bar_h),
            Qt.AlignmentFlag.AlignCenter,
            f"||Phi|| = {v:.2e}",
        )

        return y + bar_h + 6

    def _draw_coupling_ratio(self, painter: QPainter, mc: dict, y: int) -> int:
        """Draw average coupling for 8x8 (no coupling ratio section used)."""
        return y
