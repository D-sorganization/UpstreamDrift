"""Qt-specific plotting components.

This module contains the MplCanvas class for embedding matplotlib plots in PyQt6 applications.
"""

from __future__ import annotations

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    except ImportError as e:
        # Re-raise ImportError so that consumers (like plotting_core.py)
        # can catch it and use their fallback logic.
        raise ImportError("Qt backend for matplotlib not available") from e

from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for embedding in PyQt6."""

    def __init__(self, width: float = 8, height: float = 6, dpi: int = 100) -> None:
        """Initialize canvas with figure.

        Args:
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch for rendering
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
