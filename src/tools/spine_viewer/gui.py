"""PyQt6 Spine Viewer widget (SPN-15b, #12772).

Read-only: shows a hash-verified study receipt as a load heatmap. The
scientific status and ``distribution_source`` are always visible; unqualified
data is rendered in a warning colour and never labelled qualified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.spine_viewer import core

logger = get_logger(__name__)

__all__ = ["SpineViewerWidget"]

_NO_STUDY = "NO STUDY LOADED - open a study receipt (no data, nothing qualified)"
_QUALIFIED_STYLE = "background:#1b5e20;color:white;font-weight:bold;padding:6px;"
_WARN_STYLE = "background:#b71c1c;color:white;font-weight:bold;padding:6px;"
_NEUTRAL_STYLE = "background:#424242;color:white;font-weight:bold;padding:6px;"


class SpineViewerWidget(QtWidgets.QWidget):
    """Load-heatmap viewer for one spine study receipt."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._view: core.SpineStudyView | None = None
        self._figure: Figure | None = None
        self._canvas: FigureCanvasQTAgg | None = None
        self._status = QtWidgets.QLabel(_NO_STUDY)
        self._status.setWordWrap(True)
        self._status.setStyleSheet(_NEUTRAL_STYLE)
        self._source = QtWidgets.QLabel("distribution_source: -")
        self._source.setWordWrap(True)
        self._combo = QtWidgets.QComboBox()
        self._combo.addItems(core.LOAD_QUANTITIES)
        self._combo.currentTextChanged.connect(self._redraw)
        open_button = QtWidgets.QPushButton("Open Study Receipt...")
        open_button.clicked.connect(self._choose_file)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(open_button)
        controls.addWidget(QtWidgets.QLabel("Quantity:"))
        controls.addWidget(self._combo)
        controls.addStretch(1)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.addWidget(self._status)
        self._layout.addWidget(self._source)
        self._layout.addLayout(controls)

    def status_text(self) -> str:
        """Text of the always-visible status banner."""
        return self._status.text()

    def distribution_text(self) -> str:
        """Text of the always-visible distribution-source line."""
        return self._source.text()

    def current_quantity(self) -> str:
        """The load component being displayed."""
        return self._combo.currentText()

    def current_figure(self) -> Figure | None:
        """The figure on screen, or ``None`` when no valid study is loaded."""
        return self._figure

    def set_quantity(self, quantity: str) -> None:
        """Select a load component (``ValueError`` if unknown)."""
        if quantity not in core.LOAD_QUANTITIES:
            raise ValueError(f"unknown quantity {quantity!r}")
        self._combo.setCurrentText(quantity)

    def load_study(self, receipt_path: str | Path) -> None:
        """Open a receipt; any failure clears the plot and shows ERROR."""
        try:
            self._view = core.open_study(receipt_path)
        except (OSError, ValueError) as exc:
            logger.warning("spine study could not be opened: %s", exc)
            self._view = None
            self._clear_plot()
            self._status.setText(f"ERROR: study not loaded ({exc})")
            self._status.setStyleSheet(_WARN_STYLE)
            self._source.setText("distribution_source: -")
            return
        self._status.setText(core.status_banner(self._view))
        qualified = self._view.scientific_status == "qualified"
        self._status.setStyleSheet(_QUALIFIED_STYLE if qualified else _WARN_STYLE)
        self._source.setText(f"distribution_source: {self._view.distribution_source}")
        self._redraw()

    def cleanup(self) -> None:
        """Release plot resources; safe to call repeatedly."""
        self._view = None
        self._clear_plot()

    def _choose_file(self) -> None:  # pragma: no cover - modal dialog
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Study Receipt", "", "Receipt (*.json)"
        )
        if path:
            self.load_study(path)

    def _clear_plot(self) -> None:
        canvas, self._canvas = self._canvas, None
        self._figure = None
        if canvas is not None:
            self._layout.removeWidget(canvas)
            canvas.deleteLater()

    def _redraw(self, *_args: object) -> None:
        if self._view is None:
            return
        self._clear_plot()
        self._figure = core.heatmap_figure(self._view, self.current_quantity())
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._layout.addWidget(self._canvas, 1)
