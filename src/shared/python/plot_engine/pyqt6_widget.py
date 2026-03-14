"""Reusable PyQt6 PlotWidget backed by PlotSpec + MatplotlibRenderer.

Provides a drop-in QWidget containing an embedded matplotlib canvas,
toolbar with style controls, export functionality, and live theme
switching via PlotThemeManager.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .matplotlib_renderer import MatplotlibRenderer
from .specs import PlotSpec

if TYPE_CHECKING:
    from plot_theme.manager import PlotThemeManager

logger = logging.getLogger(__name__)


class PlotWidget(QWidget):
    """Reusable plot widget that renders PlotSpec via MatplotlibRenderer.

    Features:
    - Embedded matplotlib FigureCanvas with navigation toolbar
    - Export to PNG/SVG/PDF
    - Live theme switching
    - Accepts any PlotSpec subclass
    """

    spec_changed = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        theme_manager: PlotThemeManager | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme_manager = theme_manager
        self._renderer = MatplotlibRenderer(theme_manager)
        self._current_spec: PlotSpec | None = None

        self._setup_ui()

        if theme_manager:
            theme_manager.add_theme_change_callback(self._on_theme_changed)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib canvas
        self._figure = Figure(figsize=(8, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        # Control bar
        control_bar = QHBoxLayout()

        self._export_btn = QPushButton("Export")
        self._export_btn.clicked.connect(self._export_plot)

        self._format_combo = QComboBox()
        self._format_combo.addItems(["PNG", "SVG", "PDF"])

        control_bar.addWidget(self._toolbar)
        control_bar.addStretch()
        control_bar.addWidget(self._format_combo)
        control_bar.addWidget(self._export_btn)

        layout.addLayout(control_bar)
        layout.addWidget(self._canvas, stretch=1)

    def set_spec(self, spec: PlotSpec) -> None:
        """Set the plot specification and render it."""
        assert spec is not None, "spec must be provided"
        self._current_spec = spec
        self._render()
        self.spec_changed.emit()

    def get_spec(self) -> PlotSpec | None:
        """Get the current plot specification."""
        return self._current_spec

    def refresh(self) -> None:
        """Re-render the current spec (e.g., after theme change)."""
        if self._current_spec:
            self._render()

    def _render(self) -> None:
        """Render the current spec onto the canvas."""
        if self._current_spec is None:
            return

        self._figure.clear()
        self._renderer.render(self._current_spec, fig=self._figure)
        self._canvas.draw()

    def _on_theme_changed(self, _theme: Any) -> None:
        """Callback when the plot theme changes."""
        self.refresh()

    def _export_plot(self) -> None:
        """Export the current plot to a file."""
        if self._current_spec is None:
            return

        fmt = self._format_combo.currentText().lower()
        ext_map = {
            "png": "PNG Files (*.png)",
            "svg": "SVG Files (*.svg)",
            "pdf": "PDF Files (*.pdf)",
        }
        filter_str = ext_map.get(fmt, "All Files (*)")

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", f"plot.{fmt}", filter_str
        )
        if not path:
            return

        self._figure.savefig(path, format=fmt, dpi=150, bbox_inches="tight")
        logger.info(f"Plot exported to {path}")

    def get_image_bytes(self, fmt: str = "png", dpi: int = 150) -> bytes:
        """Get the current plot as image bytes."""
        assert fmt is not None, "fmt must be provided"
        if self._current_spec is None:
            return b""
        return self._renderer.to_image(self._current_spec, fmt=fmt, dpi=dpi)
