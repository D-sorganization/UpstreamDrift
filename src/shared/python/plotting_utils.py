"""Plotting utilities for eliminating matplotlib duplication.

This module provides reusable plotting patterns.

Usage:
    from src.shared.python.plotting_utils import (
        create_figure,
        save_figure,
        setup_plot_style,
    )

    fig, ax = create_figure()
    ax.plot(x, y)
    save_figure(fig, "output.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.shared.python.logging_config import get_logger
from src.shared.python.path_utils import ensure_directory

logger = get_logger(__name__)

# Default figure sizes
DEFAULT_FIGSIZE = (10, 6)
LARGE_FIGSIZE = (15, 10)
SQUARE_FIGSIZE = (8, 8)
WIDE_FIGSIZE = (12, 4)


def create_figure(
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Create matplotlib figure with consistent styling.

    Args:
        figsize: Figure size (width, height) in inches
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        **kwargs: Additional arguments for plt.subplots()

    Returns:
        Tuple of (figure, axes)

    Example:
        fig, ax = create_figure()
        fig, axes = create_figure(nrows=2, ncols=2)
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    logger.debug(f"Created figure with size {figsize}, {nrows}x{ncols} subplots")
    return fig, axes


def save_figure(
    fig: Figure,
    path: str | Path,
    dpi: int = 300,
    bbox_inches: str = "tight",
    **kwargs: Any,
) -> None:
    """Save figure with consistent settings.

    Args:
        fig: Figure to save
        path: Output path
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box setting
        **kwargs: Additional arguments for fig.savefig()

    Example:
        save_figure(fig, "output/plot.png")
        save_figure(fig, "plot.pdf", dpi=600)
    """
    path_obj = Path(path)
    ensure_directory(path_obj.parent)

    fig.savefig(path_obj, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    logger.info(f"Saved figure to {path}")


def setup_plot_style(style: str = "seaborn-v0_8-darkgrid") -> None:
    """Setup matplotlib style.

    Args:
        style: Matplotlib style name

    Example:
        setup_plot_style("seaborn-v0_8-darkgrid")
    """
    try:
        plt.style.use(style)
        logger.debug(f"Applied plot style: {style}")
    except Exception as e:
        logger.warning(f"Could not apply style {style}: {e}")


def format_axis(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    legend: bool = False,
    **kwargs: Any,
) -> None:
    """Format axis with consistent styling.

    Args:
        ax: Axes to format
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        grid: Whether to show grid
        legend: Whether to show legend
        **kwargs: Additional formatting options

    Example:
        format_axis(ax, xlabel="Time [s]", ylabel="Position [m]", title="Trajectory")
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    if legend:
        ax.legend()


def plot_time_series(
    ax: plt.Axes,
    time: np.ndarray,
    data: np.ndarray,
    label: str | None = None,
    **kwargs: Any,
) -> None:
    """Plot time series data with consistent styling.

    Args:
        ax: Axes to plot on
        time: Time array
        data: Data array
        label: Line label
        **kwargs: Additional arguments for ax.plot()

    Example:
        plot_time_series(ax, time, position, label="Position")
    """
    ax.plot(time, data, label=label, **kwargs)


def plot_multiple_time_series(
    ax: plt.Axes,
    time: np.ndarray,
    data_dict: dict[str, np.ndarray],
    **kwargs: Any,
) -> None:
    """Plot multiple time series on same axes.

    Args:
        ax: Axes to plot on
        time: Time array
        data_dict: Dictionary mapping labels to data arrays
        **kwargs: Additional arguments for ax.plot()

    Example:
        plot_multiple_time_series(
            ax,
            time,
            {"x": x_data, "y": y_data, "z": z_data}
        )
    """
    for label, data in data_dict.items():
        ax.plot(time, data, label=label, **kwargs)
    ax.legend()


def create_comparison_plot(
    time: np.ndarray,
    data1: np.ndarray,
    data2: np.ndarray,
    label1: str = "Data 1",
    label2: str = "Data 2",
    title: str = "Comparison",
    **kwargs: Any,
) -> tuple[Figure, plt.Axes]:
    """Create comparison plot with two datasets.

    Args:
        time: Time array
        data1: First dataset
        data2: Second dataset
        label1: Label for first dataset
        label2: Label for second dataset
        title: Plot title
        **kwargs: Additional arguments for create_figure()

    Returns:
        Tuple of (figure, axes)

    Example:
        fig, ax = create_comparison_plot(time, measured, simulated)
    """
    fig, ax = create_figure(**kwargs)
    ax.plot(time, data1, label=label1, linestyle="-")
    ax.plot(time, data2, label=label2, linestyle="--")
    ax.legend()
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def create_error_plot(
    time: np.ndarray,
    data1: np.ndarray,
    data2: np.ndarray,
    title: str = "Error",
    **kwargs: Any,
) -> tuple[Figure, plt.Axes]:
    """Create error plot showing difference between datasets.

    Args:
        time: Time array
        data1: First dataset
        data2: Second dataset
        title: Plot title
        **kwargs: Additional arguments for create_figure()

    Returns:
        Tuple of (figure, axes)

    Example:
        fig, ax = create_error_plot(time, measured, simulated)
    """
    fig, ax = create_figure(**kwargs)
    error = data1 - data2
    ax.plot(time, error)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_title(title)
    ax.set_ylabel("Error")
    ax.grid(True, alpha=0.3)
    return fig, ax


def create_subplot_grid(
    nrows: int,
    ncols: int,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """Create grid of subplots with consistent styling.

    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size (auto-calculated if None)
        **kwargs: Additional arguments for plt.subplots()

    Returns:
        Tuple of (figure, axes array)

    Example:
        fig, axes = create_subplot_grid(2, 3)
        for ax in axes.flat:
            ax.plot(data)
    """
    if figsize is None:
        # Auto-calculate figure size
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    logger.debug(f"Created {nrows}x{ncols} subplot grid")
    return fig, axes


def add_vertical_line(
    ax: plt.Axes,
    x: float,
    label: str | None = None,
    color: str = "r",
    linestyle: str = "--",
    **kwargs: Any,
) -> None:
    """Add vertical line to plot.

    Args:
        ax: Axes to add line to
        x: X-coordinate of line
        label: Line label
        color: Line color
        linestyle: Line style
        **kwargs: Additional arguments for ax.axvline()

    Example:
        add_vertical_line(ax, impact_time, label="Impact", color="red")
    """
    ax.axvline(x=x, color=color, linestyle=linestyle, label=label, **kwargs)


def add_horizontal_line(
    ax: plt.Axes,
    y: float,
    label: str | None = None,
    color: str = "r",
    linestyle: str = "--",
    **kwargs: Any,
) -> None:
    """Add horizontal line to plot.

    Args:
        ax: Axes to add line to
        y: Y-coordinate of line
        label: Line label
        color: Line color
        linestyle: Line style
        **kwargs: Additional arguments for ax.axhline()

    Example:
        add_horizontal_line(ax, threshold, label="Threshold")
    """
    ax.axhline(y=y, color=color, linestyle=linestyle, label=label, **kwargs)


def annotate_point(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    **kwargs: Any,
) -> None:
    """Annotate point on plot.

    Args:
        ax: Axes to annotate
        x: X-coordinate
        y: Y-coordinate
        text: Annotation text
        **kwargs: Additional arguments for ax.annotate()

    Example:
        annotate_point(ax, peak_time, peak_value, "Peak")
    """
    ax.annotate(text, xy=(x, y), **kwargs)


def close_all_figures() -> None:
    """Close all matplotlib figures.

    Example:
        close_all_figures()
    """
    plt.close("all")
    logger.debug("Closed all figures")
