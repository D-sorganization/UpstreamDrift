"""Matplotlib drawing of a Variability & Consistency :class:`PlotView` (NSX-100).

No PyQt and no physics: it draws the numbers the view carries (shown items only).
Heatmap colour limits are the grid's own extent, a display scale.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from matplotlib.figure import Figure

from src.tools.grip_wrench_plots.variability_view import PlotView, SectionView

__all__ = ["draw_sections"]


def _nan(values: Sequence[float | None]) -> list[float]:
    return [math.nan if v is None else float(v) for v in values]


def _heatmap(ax: Any, plot: PlotView) -> None:
    grid = [_nan(row) for row in plot.grid]
    image = ax.imshow(grid, aspect="auto", origin="lower")
    ax.figure.colorbar(image, ax=ax)
    ax.set_xticks(range(len(plot.labels)), plot.labels, fontsize=7)
    ax.set_yticks(range(len(plot.row_labels)), plot.row_labels, fontsize=7)


def _bars(ax: Any, plot: PlotView) -> None:
    ax.bar(plot.labels, _nan(plot.y))


def _scatter(ax: Any, plot: PlotView) -> None:
    ax.scatter(_nan(plot.x), _nan(plot.y))
    for label, x, y in zip(plot.labels, _nan(plot.x), _nan(plot.y), strict=True):
        ax.annotate(label, (x, y), fontsize=7)


def _interval(ax: Any, plot: PlotView) -> None:
    for k, (_label, lo, hi) in enumerate(
        zip(plot.labels, _nan(plot.y), _nan(plot.y2), strict=True)
    ):
        ax.vlines(k, lo, hi, linewidth=6)
        ax.plot([k], [lo], marker="_")
    ax.set_xticks(range(len(plot.labels)), plot.labels, fontsize=7)


_DRAW = {
    "heatmap": _heatmap,
    "bars": _bars,
    "scatter": _scatter,
    "interval": _interval,
}


def draw_sections(figure: Figure, sections: Sequence[SectionView]) -> None:
    """Redraw ``figure``: one axes per section that has a plot.

    Postcondition: ``len(figure.axes)`` is the number of plotted sections (plus
    colour bars); with none, the figure holds one text line.
    """
    figure.clear()
    plots = [s.plot for s in sections if s.plot is not None]
    if not plots:
        figure.text(0.5, 0.5, "nothing to draw", ha="center", va="center")
        return
    axes = figure.subplots(len(plots), 1, squeeze=False)[:, 0]
    for ax, plot in zip(axes, plots, strict=True):
        ax.set_title(plot.title, fontsize=9, loc="left")
        ax.set_xlabel(plot.xlabel, fontsize=8)
        ax.set_ylabel(plot.ylabel, fontsize=8)
        _DRAW[plot.kind](ax, plot)
    figure.tight_layout()
