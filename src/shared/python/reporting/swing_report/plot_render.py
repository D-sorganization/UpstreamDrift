"""PNG rendering of ``render=values`` plots for the package exporter (#12327).

Only plots whose descriptor says ``render == "values"`` are drawn; ``unavailable``
plots (every withheld quantity, I8) produce no file. Rendering uses the Agg
canvas directly (no pyplot, no global state) and strips volatile PNG metadata
so equal input yields equal bytes.
"""

from __future__ import annotations

import io
import logging

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from .contract import PlotDescriptor, Quantity, SwingReport

logger = logging.getLogger(__name__)

PLOTS_DIR = "plots"
_DPI = 100
_FIGSIZE = (6.4, 3.6)


def plot_path(plot: PlotDescriptor) -> str:
    """Relative package path of ``plot``'s PNG."""
    return f"{PLOTS_DIR}/{plot.plot_id}.png"


def values_plots(report: SwingReport) -> list[PlotDescriptor]:
    """Plots that render values, in report order."""
    return [p for p in report.plots if p.render == "values"]


def _array(values: tuple[float | None, ...]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _draw_band(ax, quantity: Quantity) -> None:
    unc = quantity.uncertainty
    if unc is None:
        return
    time_s = np.array(quantity.series[0].time_s, dtype=float)
    lower, upper = np.array(unc.lower), np.array(unc.upper)
    covered = np.array(
        [
            any(s.values[i] is not None for s in quantity.series)
            for i in range(len(lower))
        ]
    )
    lower[~covered] = np.nan
    upper[~covered] = np.nan
    ax.fill_between(time_s, lower, upper, alpha=0.2, label="Engine Spread")


def _draw(plot: PlotDescriptor, quantity: Quantity) -> Figure:
    fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
    ax = fig.add_subplot(1, 1, 1)
    _draw_band(ax, quantity)
    for series in quantity.series:
        label = series.engine + (f" ({series.channel})" if series.channel else "")
        ax.plot(series.time_s, _array(series.values), label=label)
    ax.set_title(plot.title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{plot.quantity_id.replace('_', ' ').title()} ({plot.units})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


def render_plot_png(plot: PlotDescriptor, quantity: Quantity) -> bytes:
    """Render ``plot`` over ``quantity`` to PNG bytes.

    Preconditions: ``plot.render == "values"`` and ``quantity`` is the plot's
    advertised quantity with at least one series.
    """
    if plot.render != "values":
        raise ValueError(f"plot {plot.plot_id!r} is not renderable ({plot.render})")
    if plot.quantity_id != quantity.quantity_id or not quantity.series:
        raise ValueError(f"plot {plot.plot_id!r} has no series to draw")
    fig = _draw(plot, quantity)
    buffer = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buffer, metadata={"Software": None})
    return buffer.getvalue()
