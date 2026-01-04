from PyQt6.QtWidgets import QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Any

class MplCanvas(FigureCanvas):
    """Matplotlib canvas widget for embedding plots in Qt."""

    def __init__(
        self,
        parent: QWidget | None = None,
        width: float = 5.0,
        height: float = 4.0,
        dpi: int = 100,
    ) -> None:
        """Initialize the matplotlib canvas with specified dimensions."""
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)  # type: ignore
        self.setParent(parent)

    def clear_axes(self) -> None:
        """Clear all axes from the figure."""
        self.fig.clear()
        self.draw()  # type: ignore

    def add_subplot(self, *args: Any, **kwargs: Any) -> Axes:
        """Add a subplot to the figure and return the axes."""
        ax = self.fig.add_subplot(*args, **kwargs)
        return ax
