"""Base class for plotting renderers.

Provides shared axis formatting, grid styling, and legend defaults
to eliminate duplicated matplotlib boilerplate across all renderers.
"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.shared.python.plotting.config import COLORS
from src.shared.python.plotting.transforms import DataManager

# ─── Shared Constants ───────────────────────────────────────────

DEFAULT_LABEL_FONTSIZE = 12
DEFAULT_TITLE_FONTSIZE = 14
DEFAULT_LABEL_FONTWEIGHT = "bold"
DEFAULT_GRID_ALPHA = 0.3
DEFAULT_GRID_STYLE = "--"
DEFAULT_LEGEND_ALPHA = 0.9


class BaseRenderer:
    """Base class for all plot renderers.

    Provides:
    - Shared color palette from config
    - Standardized axis formatting (labels, titles, grid, legend)
    - Figure clearing
    """

    def __init__(self, data_manager: DataManager) -> None:
        """Initialize renderer with data manager."""
        self.data = data_manager
        self.colors = COLORS

    def clear_figure(self, fig: Figure) -> None:
        """Clear the figure before plotting."""
        fig.clear()

    @staticmethod
    def format_axis(
        ax: Axes,
        *,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        grid: bool = True,
        legend: bool = True,
        legend_loc: str = "best",
        **kwargs: Any,
    ) -> None:
        """Apply standardized formatting to a matplotlib axis.

        This eliminates duplicated fontsize/fontweight/grid/legend
        boilerplate across all renderers.

        Args:
            ax: The matplotlib axis to format.
            xlabel: X-axis label text.
            ylabel: Y-axis label text.
            title: Plot title text.
            grid: Whether to show grid lines.
            legend: Whether to show legend.
            legend_loc: Legend location string.
            **kwargs: Additional axis configuration.
        """
        if xlabel:
            ax.set_xlabel(
                xlabel,
                fontsize=kwargs.get("label_fontsize", DEFAULT_LABEL_FONTSIZE),
                fontweight=kwargs.get("label_fontweight", DEFAULT_LABEL_FONTWEIGHT),
            )
        if ylabel:
            ax.set_ylabel(
                ylabel,
                fontsize=kwargs.get("label_fontsize", DEFAULT_LABEL_FONTSIZE),
                fontweight=kwargs.get("label_fontweight", DEFAULT_LABEL_FONTWEIGHT),
            )
        if title:
            ax.set_title(
                title,
                fontsize=kwargs.get("title_fontsize", DEFAULT_TITLE_FONTSIZE),
                fontweight=kwargs.get("title_fontweight", DEFAULT_LABEL_FONTWEIGHT),
            )
        if grid:
            ax.grid(
                True,
                alpha=kwargs.get("grid_alpha", DEFAULT_GRID_ALPHA),
                linestyle=kwargs.get("grid_style", DEFAULT_GRID_STYLE),
            )
        if legend and ax.get_legend_handles_labels()[1]:
            ax.legend(
                loc=legend_loc,
                framealpha=kwargs.get("legend_alpha", DEFAULT_LEGEND_ALPHA),
            )
