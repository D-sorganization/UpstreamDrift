"""Base class for plotting renderers."""

from __future__ import annotations

from matplotlib.figure import Figure

from src.shared.python.plotting.config import COLORS
from src.shared.python.plotting.transforms import DataManager


class BaseRenderer:
    """Base class for all plot renderers."""

    def __init__(self, data_manager: DataManager) -> None:
        """Initialize renderer with data manager."""
        self.data = data_manager
        self.colors = COLORS

    def clear_figure(self, fig: Figure) -> None:
        """Clear the figure before plotting."""
        fig.clear()
