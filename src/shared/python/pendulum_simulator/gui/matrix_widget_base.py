"""
Base class for matrix visualization widgets in pendulum simulations.

Provides common rendering infrastructure for mass matrix display, force balance,
and energy visualization. Subclasses implement model-specific matrix sizes and
coupling ratio calculations.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


class MatrixWidgetBase(QWidget):
    """Abstract base class for matrix visualization widgets.

    Shared responsibilities:
      - Layout and simulation state management
      - paintEvent orchestration (empty state, section drawing)
      - Common drawing helpers (titles, coupling bars, force/energy tables)
      - Color scheme constants

    Subclasses must implement:
      - get_matrix_size(): Return (rows, cols) for the mass matrix
      - get_matrix_entries(mc: dict): Return list of (row, col, label, value, is_diag)
      - get_column_labels(): Return list of column DOF labels
      - _draw_coupling_ratio(painter, mc, y): Model-specific coupling calculation
    """

    # Shared color palette
    COLOR_DIAGONAL = QColor(70, 130, 230)
    COLOR_OFFDIAG = QColor(230, 160, 50)
    COLOR_BG = QColor(30, 30, 40)
    COLOR_CELL_BG_DIAG = QColor(40, 55, 80)
    COLOR_CELL_BG_OFF = QColor(80, 60, 30)
    COLOR_TEXT = QColor(220, 220, 235)
    COLOR_LABEL = QColor(160, 160, 180)
    COLOR_BRACKET = QColor(120, 120, 140)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the matrix widget base.

        Pre:
          - parent is None or a valid QWidget
        Post:
          - Widget is ready for simulation data
        """
        super().__init__(parent)
        self.setMinimumSize(200, 300)
        self._result: Any = None
        self._current_idx: int = 0
        logger.debug("%s initialized", self.__class__.__name__)

    def set_simulation(self, result: Any) -> None:
        """Set the simulation result to display.

        Pre:
          - result is not None
          - result has n_steps >= 1
        """
        assert result is not None, f"{self.__class__.__name__}: result must not be None"
        assert (
            result.n_steps >= 1
        ), f"{self.__class__.__name__}: result must have at least one time step"
        self._result = result
        self._current_idx = 0
        logger.debug(
            "%s: simulation set with %d steps",
            self.__class__.__name__,
            result.n_steps,
        )
        self.update()

    def set_frame(self, idx: int) -> None:
        """Set the current frame index.

        Pre:
          - idx >= 0
        Post:
          - _current_idx is clamped to [0, n_steps-1]
        """
        assert idx is not None, "idx must be provided"
        if self._result is None:
            return
        self._current_idx = max(0, min(idx, self._result.n_steps - 1))
        self.update()

    def clear(self) -> None:
        """Clear the displayed simulation."""
        self._result = None
        self._current_idx = 0
        logger.debug("%s: simulation cleared", self.__class__.__name__)
        self.update()

    def paintEvent(self, event: object) -> None:
        """Render the widget. Orchestrates sections and delegates to helpers."""
        assert event is not None, "event must be provided"
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.COLOR_BG)

        if self._result is None:
            painter.setPen(self.COLOR_LABEL)
            painter.setFont(QFont("Sans", 11))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No simulation loaded"
            )
            painter.end()
            return

        mc = self._result.mass_matrix_at(self._current_idx)
        y_cursor = 10

        y_cursor = self._draw_section_title(painter, "Mass Matrix M(q)", y_cursor)
        y_cursor = self._draw_matrix(painter, mc, y_cursor)
        y_cursor += 10

        y_cursor = self._draw_section_title(painter, "Coupling Ratio", y_cursor)
        y_cursor = self._draw_coupling_ratio(painter, mc, y_cursor)
        y_cursor += 10

        y_cursor = self._draw_section_title(painter, "Force Balance", y_cursor)
        y_cursor = self._draw_force_balance(painter, y_cursor)
        y_cursor += 10

        y_cursor = self._draw_section_title(painter, "Energy", y_cursor)
        y_cursor = self._draw_energy(painter, y_cursor)

        painter.end()

    # ======================================================================
    # Abstract methods (subclass-specific)
    # ======================================================================

    @abstractmethod
    def get_matrix_size(self) -> tuple[int, int]:
        """Return (rows, cols) for the mass matrix grid layout.

        Returns
        -------
        tuple[int, int]
            Tuple of (num_rows, num_cols), e.g., (2, 2), (3, 3), (8, 8)
        """

    @abstractmethod
    def get_matrix_entries(self, mc: dict) -> list:
        """Build list of matrix cell entries for rendering.

        Parameters
        ----------
        mc : dict
            Dictionary with matrix values

        Returns
        -------
        list
            List of (row, col, label, value, is_diag) tuples, one per cell
        """

    @abstractmethod
    def get_column_labels(self) -> list[str]:
        """Return list of DOF (degree of freedom) column labels.

        Returns
        -------
        list[str]
            List of label strings, or empty list if no labels needed
        """

    @abstractmethod
    def _draw_coupling_ratio(self, painter: QPainter, mc: dict, y: int) -> int:
        """Draw the coupling ratio bar. Subclass computes coupling metric.

        Parameters
        ----------
        painter : QPainter
            Valid painter object drawing to this widget
        mc : dict
            Current mass matrix dictionary
        y : int
            Current cursor position

        Returns
        -------
        int
            New y cursor position after drawing
        """

    # ======================================================================
    # Shared drawing helpers
    # ======================================================================

    def _draw_section_title(self, painter: QPainter, title: str, y: int) -> int:
        """Draw a section title with underline.

        Pre:
          - painter is valid
          - title is non-empty
          - y >= 0
        Returns:
            New y cursor position
        """
        assert painter is not None, "painter must be provided"
        painter.setPen(self.COLOR_TEXT)
        font = QFont("Sans", 11, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(15, y + 16, title)
        # underline
        painter.setPen(QPen(self.COLOR_BRACKET, 1))
        painter.drawLine(15, y + 20, self.width() - 15, y + 20)
        return y + 28

    def _draw_matrix(self, painter: QPainter, mc: dict, y: int) -> int:
        """Draw the mass matrix grid with color-coded cells.

        Pre:
          - painter is valid
          - mc contains all matrix values keyed by entry labels (M11, M12, etc.)
          - y >= 0
        Returns:
            New y cursor position after matrix and legend
        """
        assert painter is not None, "painter must be provided"
        rows, cols = self.get_matrix_size()
        entries = self.get_matrix_entries(mc)
        _ = self.get_column_labels()  # reserved for future DOF labels in matrix

        # Compute cell dimensions to fit matrix
        cell_w = max(80, (self.width() - 60) // cols)
        cell_h = max(40, cell_w)
        gap = 8 if rows > 2 else 10
        grid_w = cols * cell_w + (cols - 1) * gap
        margin_x = (self.width() - grid_w) // 2
        bracket_w = 8

        # Bracket coordinates
        bx_left = margin_x - bracket_w - 4
        bx_right = margin_x + grid_w + bracket_w + 4
        by_top = y
        by_bot = y + rows * cell_h + (rows - 1) * gap

        # Draw brackets
        painter.setPen(QPen(self.COLOR_BRACKET, 2))
        painter.drawLine(bx_left + bracket_w, by_top, bx_left, by_top)
        painter.drawLine(bx_left, by_top, bx_left, by_bot)
        painter.drawLine(bx_left, by_bot, bx_left + bracket_w, by_bot)
        painter.drawLine(bx_right - bracket_w, by_top, bx_right, by_top)
        painter.drawLine(bx_right, by_top, bx_right, by_bot)
        painter.drawLine(bx_right, by_bot, bx_right - bracket_w, by_bot)

        # Draw cells
        for row, col, label, value, is_diag in entries:
            cx = margin_x + col * (cell_w + gap)
            cy = y + row * (cell_h + gap)

            bg = self.COLOR_CELL_BG_DIAG if is_diag else self.COLOR_CELL_BG_OFF
            border = self.COLOR_DIAGONAL if is_diag else self.COLOR_OFFDIAG

            rect = QRectF(cx, cy, cell_w, cell_h)
            painter.setPen(QPen(border, 2))
            painter.setBrush(QBrush(bg))
            painter.drawRoundedRect(rect, 6, 6)

            # Label
            painter.setPen(self.COLOR_LABEL)
            font_label = QFont("Monospace", 8)
            painter.setFont(font_label)
            label_y = cy + int(cell_h * 0.35)
            painter.drawText(cx + 5, label_y, label)

            # Value
            painter.setPen(self.COLOR_TEXT)
            font_val = QFont("Monospace", 10 if rows > 2 else 14, QFont.Weight.Bold)
            painter.setFont(font_val)
            value_y = cy + int(cell_h * 0.8)
            painter.drawText(cx + 5, value_y, f"{value:.3f}")

        # Legend
        ly = y + rows * cell_h + (rows - 1) * gap + 16
        painter.setFont(QFont("Sans", 9))

        painter.setPen(self.COLOR_DIAGONAL)
        painter.drawText(margin_x, ly, "\u25a0 Diagonal (self-coupling)")

        painter.setPen(self.COLOR_OFFDIAG)
        painter.drawText(margin_x, ly + 16, "\u25a0 Off-diagonal (cross-coupling)")

        return ly + 32

    def _draw_force_balance(self, painter: QPainter, y: int) -> int:
        """Draw torque, Coriolis, and gravity vectors.

        Pre:
          - _result is not None (checked in paintEvent)
          - painter is valid
          - y >= 0
        Returns:
            New y cursor position
        """
        assert self._result is not None
        idx = self._current_idx
        tau = self._result.torques_at(idx)
        G = self._result.gravity_at(idx)

        painter.setFont(QFont("Monospace", 9 if len(tau) > 2 else 10))
        lines = []

        # Tau lines (use joint names if available, else generic indices)
        col_labels = self.get_column_labels()
        for i, t in enumerate(tau):
            if i < len(col_labels):
                label = col_labels[i]
            else:
                label = f"joint_{i + 1}"
            lines.append((f"tau_{label:>4s} = {t:+8.2f} N*m", self.COLOR_TEXT))

        # Gravity lines
        for i, g in enumerate(G):
            if i < len(col_labels):
                label = col_labels[i]
            else:
                label = f"joint_{i + 1}"
            lines.append((f"G_{label:>5s} = {g:+8.2f} N*m", QColor(130, 200, 130)))

        # Try to add Coriolis if available
        try:
            C = self._result.coriolis_at(idx)
            for i, c in enumerate(C):
                if i < len(col_labels):
                    label = col_labels[i]
                else:
                    label = f"joint_{i + 1}"
                lines.append((f"C_{label:>5s} = {c:+8.2f} N*m", QColor(180, 130, 200)))
        except (AttributeError, TypeError):
            pass

        for text, color in lines:
            painter.setPen(color)
            painter.drawText(20, y + 14, text)
            y += 16

        return y + 4

    def _draw_energy(self, painter: QPainter, y: int) -> int:
        """Draw kinetic, potential, and total energy.

        Pre:
          - _result is not None
          - painter is valid
          - y >= 0
        Returns:
            New y cursor position
        """
        assert self._result is not None
        e = self._result.energy_at(self._current_idx)
        painter.setFont(QFont("Monospace", 10))
        lines = [
            (f"Kinetic    = {e['kinetic']:+8.2f} J", QColor(230, 160, 80)),
            (f"Potential  = {e['potential']:+8.2f} J", QColor(80, 180, 230)),
            (f"Total      = {e['total']:+8.2f} J", self.COLOR_TEXT),
        ]
        for text, color in lines:
            painter.setPen(color)
            painter.drawText(20, y + 14, text)
            y += 18
        return y + 4
