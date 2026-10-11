"""Ground-reaction plot sheet (GCV-5, #11711).

One matplotlib figure, shared by the PyQt widget and the report/video export,
drawn from a :class:`GroundReactionPlotSeries`: vertical force per foot and net
(in body weights when known), the net force components, the vertical load
share, the centre-of-pressure path in top view, the vertical free moment and
the net moment about the centre of mass.  Unavailable samples leave gaps; a
panel without data says "unavailable" with the reason.
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from src.shared.python.biomechanics.ground_reaction_plot_model import (
    GroundReactionPlotSeries,
)
from src.shared.python.plotting.renderers.time_panels import (
    draw_time_panel,
    mark_unavailable,
)

__all__ = ["PANEL_TITLES", "plot_ground_reaction"]

PANEL_TITLES: tuple[str, ...] = (
    "Vertical Ground Reaction Force",
    "Net Force Components",
    "Vertical Load Share",
    "Centre of Pressure Path",
    "Free Moment About the Vertical",
    "Net Moment About CoM",
)

_KEY_COLORS = ("#0072B2", "#E69F00", "#CC79A7", "#56B4E9")
_NET_COLOR = "#009E73"
_AXIS_ROWS = (
    ("x", "x", "#D55E00"),
    ("y", "y", "#009E73"),
    ("z", "z", "#0072B2"),
    ("magnitude", "|F|", "#000000"),
)


def _values(raw: list[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in raw], dtype=float)


def _component(
    series: GroundReactionPlotSeries, trace: str, axis: str
) -> np.ndarray | None:
    return _values(series.traces[trace][axis]) if trace in series.traces else None


def _keys(series: GroundReactionPlotSeries) -> list[tuple[str, str, str]]:
    """``(key, legend label, colour)`` per foot, then net."""
    feet = [
        (k, k.replace("_", " ").title(), _KEY_COLORS[i % len(_KEY_COLORS)])
        for i, k in enumerate(series.feet)
    ]
    return [*feet, ("net", "Net", _NET_COLOR)] if series.traces else []


_Row = tuple[np.ndarray, str, str]


def _per_key_rows(
    series: GroundReactionPlotSeries, quantity: str, axis: str
) -> list[_Row]:
    rows: list[_Row] = []
    for key, label, color in _keys(series):
        values = _component(series, f"{key}_{quantity}", axis)
        if values is not None:
            rows.append((values, label, color))
    return rows


def _axis_rows(
    series: GroundReactionPlotSeries,
    trace: str,
    axes: tuple[tuple[str, str, str], ...],
) -> list[_Row]:
    """One row per component of ``trace``; empty when the trace is absent."""
    rows: list[_Row] = []
    for axis, label, color in axes:
        values = _component(series, trace, axis)
        if values is not None:
            rows.append((values, label, color))
    return rows


def _cop_path(ax: Axes, series: GroundReactionPlotSeries) -> None:
    ax.set_title(PANEL_TITLES[3], fontsize=9)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    shown = 0
    for key, label, color in _keys(series):
        x = _component(series, f"{key}_cop_m", "x")
        y = _component(series, f"{key}_cop_m", "y")
        if x is None or y is None:
            continue
        shown += int(np.isfinite(x).any())
        ax.plot(x, y, color=color, linewidth=1.4, label=label)
    if shown:
        ax.legend(fontsize=7, loc="upper left")
    else:
        mark_unavailable(ax, series.reason)
    ax.tick_params(labelsize=7)


def plot_ground_reaction(fig: Figure, series: GroundReactionPlotSeries) -> None:
    """Draw the six ground-reaction panels onto ``fig`` (cleared first).

    Raises:
        TypeError: if ``series`` is not a :class:`GroundReactionPlotSeries`.
    """
    if not isinstance(series, GroundReactionPlotSeries):
        raise TypeError(f"series must be a GroundReactionPlotSeries: {type(series)}")
    fig.clear()
    axes = fig.subplots(3, 2).ravel()
    time_s, events, reason = series.time_s, series.events, series.reason
    in_bw = "net_force_bw" in series.traces
    force = "force_bw" if in_bw else "force_n"
    net_force = _axis_rows(series, "net_force_n", _AXIS_ROWS)
    net_moment = _axis_rows(series, "net_moment_com_nm", _AXIS_ROWS[:3])
    share = [
        (_values(series.load_share[key]), label, color)
        for key, label, color in _keys(series)
        if key in series.load_share
    ]
    panels = {
        0: (_per_key_rows(series, force, "z"), "BW" if in_bw else "N"),
        1: (net_force, "N"),
        2: (share, "fraction"),
        4: (_per_key_rows(series, "free_moment_nm", "z"), "N*m"),
        5: (net_moment, "N*m"),
    }
    for index, (rows, unit) in panels.items():
        draw_time_panel(
            axes[index],
            time_s,
            rows,
            title=PANEL_TITLES[index],
            unit=unit,
            events=events,
            reason=reason,
        )
    _cop_path(axes[3], series)
    for ax in axes[4:]:
        ax.set_xlabel("Time (s)", fontsize=8)
    fig.suptitle("Ground Reaction (on the body, world axes)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
