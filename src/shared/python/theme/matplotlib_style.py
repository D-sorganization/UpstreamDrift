"""Professional matplotlib styling for the repository fleet.

This module provides a consistent, sophisticated visual style for all
matplotlib plots across the application, integrated with the theme system.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt

from .colors import CHART_COLORS

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def apply_plot_theme(fig: Figure, theme: dict[str, str]) -> None:
    """Apply a theme to a matplotlib figure.

    Args:
        fig: Matplotlib Figure to style
        theme: Theme dictionary containing color definitions
    """
    # Style figure
    fig.set_facecolor(theme.get("bg", "#ffffff"))

    # Style axes
    for ax in fig.get_axes():
        ax.set_facecolor(theme.get("group_bg", "#f8f9fa"))

        # Style spines
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["bottom", "left"]:
            ax.spines[side].set_color(theme.get("border", "#ced4da"))
            ax.spines[side].set_linewidth(1.0)

        # Style ticks and labels
        text_color = theme.get("text", "#212529")
        secondary_text = theme.get("text_secondary", "#495057")

        ax.tick_params(colors=secondary_text, which="both", labelsize=9)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)

        # Style grid
        ax.grid(
            True,
            color=theme.get("border", "#ced4da"),
            linestyle="-",
            linewidth=0.5,
            alpha=0.3,
        )
        ax.set_axisbelow(True)

        # Style legend
        legend = ax.get_legend()
        if legend:
            frame = legend.get_frame()
            frame.set_facecolor(theme.get("group_bg", "#f8f9fa"))
            frame.set_edgecolor(theme.get("border", "#ced4da"))
            frame.set_alpha(0.9)
            for text in legend.get_texts():
                text.set_color(text_color)

    if hasattr(fig, "canvas") and fig.canvas:
        fig.canvas.draw_idle()


def apply_global_style() -> None:
    """Apply global matplotlib defaults for new plots.

    This sets up the global rcParams that don't depend on the specific theme colors
    but rather on the general aesthetic (cycle, fonts, etc).
    """
    mpl.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler("color", CHART_COLORS),
            "axes.grid": True,
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "font.size": 10,
            "figure.autolayout": True,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def get_chart_color(index: int) -> str:
    """Get a chart color by index (cycles through palette)."""
    return CHART_COLORS[index % len(CHART_COLORS)]


def create_styled_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Create a pre-styled figure with the current global theme defaults."""
    if figsize is None:
        figsize = (10, 6)

    apply_global_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    # Note: caller should call apply_plot_theme(fig, current_theme)
    # to get full theme-aware colors.

    return fig, axes


__all__ = [
    "apply_plot_theme",
    "apply_global_style",
    "create_styled_figure",
    "get_chart_color",
]
