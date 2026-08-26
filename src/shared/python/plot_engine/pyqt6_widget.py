# mypy: ignore-errors
"""Reusable PyQt6 PlotWidget backed by PlotSpec + MatplotlibRenderer.

Provides a drop-in QWidget containing an embedded matplotlib canvas,
toolbar with style controls, export functionality, and live theme
switching via PlotThemeManager.
"""

from __future__ import annotations

import logging
from pathlib import Path
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

from src.shared.python.plotting.export import ExportConfig, export_figure
from src.shared.python.plotting.identity import PlotIdentity
from src.shared.python.theme.integration import get_theme_manager
from src.shared.python.theme.matplotlib_style import apply_plot_theme

from .matplotlib_renderer import MatplotlibRenderer
from .specs import PlotSpec

if TYPE_CHECKING:
    from shared.python.plot_theme.manager import PlotThemeManager

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
        self._identity: PlotIdentity | None = None

        self._setup_ui()

        if theme_manager:
            theme_manager.add_theme_change_callback(self._on_theme_changed)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib canvas
        self._figure = Figure(figsize=(8, 6), dpi=100)
        _tm = get_theme_manager()
        apply_plot_theme(self._figure, _tm.get_current_colors())
        _tm.themeChanged.connect(
            lambda name: apply_plot_theme(
                self._figure, _tm.get_theme_colors(name) or _tm.get_current_colors()
            )
        )
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
        if spec is None:
            raise ValueError("spec must be provided")
        self._current_spec = spec
        self._render()
        self.spec_changed.emit()

    def get_spec(self) -> PlotSpec | None:
        """Get the current plot specification."""
        return self._current_spec

    def set_identity(self, identity: PlotIdentity | None) -> None:
        """Attach engine/model/run identity used for export metadata.

        Optional: callers that know which physics engine/model/run
        produced the plotted data can call this so exports carry that
        provenance (see ``PlotIdentity``). Defaults to ``None`` (no
        identity fields embedded beyond the export timestamp).
        """
        self._identity = identity

    def get_identity(self) -> PlotIdentity | None:
        """Return the identity currently attached to this widget's exports."""
        return self._identity

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
        """Export the current plot to a file, embedding export metadata."""
        if self._current_spec is None:
            return

        fmt = self._format_combo.currentText().lower()
        ext_map = {
            "png": "PNG Files (*.png)",
            "svg": "SVG Files (*.svg)",
            "pdf": "PDF Files (*.pdf)",
        }
        filter_str = ext_map.get(fmt, "All Files (*)")

        default_name = self._current_spec.title or "plot"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", f"{default_name}.{fmt}", filter_str
        )
        if not path:
            return

        save_path = Path(path)
        export_config = ExportConfig(
            output_dir=save_path.parent,
            dpi=150,
            bbox_inches="tight",
            include_metadata=True,
        )
        export_figure(
            self._figure,
            save_path.stem,
            config=export_config,
            formats=[fmt],
            identity=self._identity,
        )
        logger.info(f"Plot exported to {save_path}")

    def get_image_bytes(self, fmt: str = "png", dpi: int = 150) -> bytes:
        """Get the current plot as image bytes."""
        if fmt is None:
            raise ValueError("fmt must be provided")
        if self._current_spec is None:
            return b""
        return self._renderer.to_image(self._current_spec, fmt=fmt, dpi=dpi)
