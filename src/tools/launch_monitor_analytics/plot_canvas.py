"""Small reusable matplotlib canvas widget for the Launch Monitor Analytics workbench."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtWidgets

__all__ = ["PlotCanvas"]


class PlotCanvas(FigureCanvasQTAgg):
    """Small reusable matplotlib canvas."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(6.0, 4.0), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.empty("Import data to begin.")

    def empty(self, message: str) -> None:
        """Fully reset the canvas (dropping any colorbar axes) and show `message`.

        Uses ``reset_axes`` rather than clearing the current axes in place
        so that artifacts from a prior analysis -- most notably a
        colorbar's own axes -- cannot survive onto the placeholder state.
        """
        axes = self.reset_axes()
        axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes)
        axes.set_axis_off()
        self.draw_idle()

    def reset_axes(self) -> Axes:
        """Replace every axes, including colorbars from a prior analysis."""
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        return self.axes
