"""Professional matplotlib styling for Golf Modeling Suite.

This module provides a consistent, sophisticated visual style for all
matplotlib plots across the application.

Usage:
    from shared.python.theme.matplotlib_style import apply_golf_suite_style

    # Apply globally at application startup
    apply_golf_suite_style()

    # Or apply to a specific figure
    apply_golf_suite_style(fig)
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from .colors import CHART_COLORS, Colors

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Golf Suite matplotlib style dictionary
GOLF_SUITE_STYLE = {
    # Figure
    "figure.facecolor": Colors.BG_BASE,
    "figure.edgecolor": Colors.BORDER_DEFAULT,
    "figure.dpi": 100,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    # Axes
    "axes.facecolor": Colors.BG_SURFACE,
    "axes.edgecolor": Colors.BORDER_DEFAULT,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.grid.axis": "both",
    "axes.axisbelow": True,  # Grid behind data
    "axes.labelcolor": Colors.TEXT_SECONDARY,
    "axes.labelsize": 10,
    "axes.labelweight": "normal",
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlecolor": Colors.TEXT_PRIMARY,
    "axes.titlepad": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": plt.cycler("color", CHART_COLORS),
    # Grid
    "grid.color": Colors.GRID_LINE,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    # Ticks
    "xtick.color": Colors.TICK_COLOR,
    "ytick.color": Colors.TICK_COLOR,
    "xtick.labelcolor": Colors.TEXT_TERTIARY,
    "ytick.labelcolor": Colors.TEXT_TERTIARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    # Legend
    "legend.facecolor": Colors.BG_ELEVATED,
    "legend.edgecolor": Colors.BORDER_DEFAULT,
    "legend.labelcolor": Colors.TEXT_SECONDARY,
    "legend.fontsize": 9,
    "legend.framealpha": 0.95,
    "legend.borderpad": 0.5,
    "legend.labelspacing": 0.4,
    "legend.handlelength": 1.5,
    "legend.handleheight": 0.7,
    # Text
    "text.color": Colors.TEXT_SECONDARY,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Inter",
        "SF Pro Display",
        "Segoe UI",
        "Roboto",
        "Helvetica Neue",
        "DejaVu Sans",
        "sans-serif",
    ],
    "font.size": 10,
    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "lines.markeredgewidth": 0.5,
    # Patches (bars, histograms, etc.)
    "patch.linewidth": 0.5,
    "patch.edgecolor": Colors.BORDER_SUBTLE,
    # Scatter
    "scatter.edgecolors": "none",
    # Savefig
    "savefig.facecolor": Colors.BG_BASE,
    "savefig.edgecolor": "none",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}


def apply_golf_suite_style(fig: "Figure | None" = None) -> None:
    """Apply the Golf Suite professional style to matplotlib.

    Args:
        fig: Optional specific figure to style. If None, applies globally.

    Example:
        # Global application (at startup)
        apply_golf_suite_style()

        # Or style a specific figure
        fig, ax = plt.subplots()
        apply_golf_suite_style(fig)
    """
    if fig is None:
        # Apply globally
        plt.rcParams.update(GOLF_SUITE_STYLE)
    else:
        # Apply to specific figure
        fig.set_facecolor(Colors.BG_BASE)
        for ax in fig.get_axes():
            _style_axes(ax)


def _style_axes(ax: "plt.Axes") -> None:
    """Apply styling to a specific axes object.

    Args:
        ax: Matplotlib Axes to style
    """
    ax.set_facecolor(Colors.BG_SURFACE)

    # Style spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(Colors.BORDER_DEFAULT)
        ax.spines[spine].set_linewidth(1.0)

    # Style grid
    ax.grid(True, color=Colors.GRID_LINE, linestyle="-", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    # Style ticks
    ax.tick_params(
        colors=Colors.TICK_COLOR,
        labelcolor=Colors.TEXT_TERTIARY,
        direction="out",
    )


def get_chart_color(index: int) -> str:
    """Get a chart color by index (cycles through palette).

    Args:
        index: Color index

    Returns:
        Hex color string
    """
    return CHART_COLORS[index % len(CHART_COLORS)]


def create_styled_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> tuple["Figure", "plt.Axes"]:
    """Create a pre-styled figure with the Golf Suite theme.

    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size in inches (width, height)
        **kwargs: Additional arguments passed to plt.subplots

    Returns:
        Tuple of (Figure, Axes or array of Axes)
    """
    if figsize is None:
        figsize = (10, 6)

    # Ensure style is applied
    apply_golf_suite_style()

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    # Apply additional styling
    fig.set_facecolor(Colors.BG_BASE)

    return fig, axes


def style_for_export(fig: "Figure", dpi: int = 200) -> None:
    """Prepare a figure for high-quality export.

    Args:
        fig: Figure to prepare
        dpi: DPI for export (default 200 for crisp output)
    """
    fig.set_dpi(dpi)
    fig.set_facecolor(Colors.BG_BASE)

    # Ensure tight layout
    fig.tight_layout()


__all__ = [
    "GOLF_SUITE_STYLE",
    "apply_golf_suite_style",
    "create_styled_figure",
    "get_chart_color",
    "style_for_export",
]
