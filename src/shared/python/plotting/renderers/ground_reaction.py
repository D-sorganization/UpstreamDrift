"""Ground-reaction plot sheet (GCV-5, #11711).

One matplotlib figure, shared by the PyQt widget and the report/video export,
drawn from a :class:`GroundReactionPlotSeries`: vertical force per foot and net
(in body weights when known), the net force components, the vertical load
share, the centre-of-pressure path in top view, the vertical free moment and
the net moment about the centre of mass.  Unavailable samples leave gaps; a
panel without data says "unavailable" with the reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
import numpy as np

from src.shared.python.biomechanics.ground_reaction_plot_model import (
    GroundReactionPlotSeries,
)
from src.shared.python.plotting.renderers.time_panels import (
    draw_time_panel,
    mark_unavailable,
)

__all__ = [
    "MAX_COP_MARKERS",
    "PANEL_TITLES",
    "SIGMA_BAND_SUFFIX",
    "cop_marker_indices",
    "UNQUALIFIED_BANNER",
    "plot_ground_reaction",
]

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


#: Banner shown whenever the overlaid plate trial has no device qualification.
UNQUALIFIED_BANNER = "Unqualified measurement"
_MEASURED_STYLE: dict[str, Any] = {"linestyle": "--", "linewidth": 1.4, "alpha": 0.9}
#: Legend suffix of the shaded 1-sigma band around a measured load.
SIGMA_BAND_SUFFIX = "\u00b11\u03c3"
_BAND_ALPHA = 0.2
#: CoP uncertainty ellipses drawn per foot at most, so the path stays legible.
MAX_COP_MARKERS = 12
#: The CoP sigma is one sigma per horizontal axis, not a 2-D covariance.
_PER_AXIS = "per axis"


def _measured_keys(series: GroundReactionPlotSeries) -> list[tuple[str, str, str]]:
    return [
        (f"measured_{key}", f"Measured {label}", color)
        for key, label, color in _keys(series)
        if f"measured_{key}_force_n" in series.traces
        or f"measured_{key}_cop_m" in series.traces
    ]


def cop_marker_indices(
    cop_x: Sequence[float | None],
    sigma: Sequence[float | None],
    max_markers: int,
) -> list[int]:
    """Evenly spaced frames, at most ``max_markers``, with both a CoP and a sigma.

    Raises ``ValueError`` for a non-positive ``max_markers`` or unequal lengths.
    """
    if max_markers < 1:
        raise ValueError("max_markers must be positive")
    if len(cop_x) != len(sigma):
        raise ValueError("cop_x and sigma must have equal lengths")
    usable = [
        i
        for i, (x, s) in enumerate(zip(cop_x, sigma, strict=True))
        if x is not None and s is not None and math.isfinite(x) and math.isfinite(s)
    ]
    stride = max(1, math.ceil(len(usable) / max_markers))
    return usable[::stride]


def _semi_axes(sigma: Mapping[str, list[float | None]], i: int) -> tuple[float, float]:
    """Per-world-axis sigma at frame ``i``, else the worse axis on both."""
    worse = sigma["z"][i]
    assert worse is not None, "cop_marker_indices keeps only frames with a sigma"
    sx, sy = sigma["x"][i], sigma["y"][i]
    return (worse, worse) if sx is None or sy is None else (sx, sy)


def _draw_cop_ellipses(
    ax: Axes, series: GroundReactionPlotSeries, key: str, style: tuple[str, str]
) -> bool:
    """Axis-aligned 1-sigma ellipses at sampled CoP points.

    The semi-axes are the per-world-axis sigmas (a circle of the worse-axis
    sigma when a payload has only that).  False without a CoP sigma trace.
    """
    label, color = style
    cop = series.traces.get(f"{key}_cop_m")
    sigma = series.traces.get(f"{key}_cop_sigma_m")
    if cop is None or sigma is None:
        return False
    rings = [
        ((x, y), _semi_axes(sigma, i))
        for i in cop_marker_indices(cop["x"], sigma["z"], MAX_COP_MARKERS)
        if (x := cop["x"][i]) is not None and (y := cop["y"][i]) is not None
    ]
    if not rings:
        return False
    # One legend entry per foot: only the first ellipse carries the label.
    names = [f"{label} {SIGMA_BAND_SUFFIX} {_PER_AXIS}"] + ["_nolegend_"] * len(rings)
    for (center, (sx, sy)), name in zip(rings, names, strict=False):
        ax.add_patch(
            Ellipse(
                center,
                2.0 * sx,
                2.0 * sy,
                facecolor=color,
                edgecolor=color,
                alpha=_BAND_ALPHA,
                label=name,
            )
        )
    return True


def _draw_band(
    ax: Axes, series: GroundReactionPlotSeries, key: str, label: str, color: str
) -> bool:
    """Shade ``load +/- sigma`` for ``key``; False when it has no sigma trace."""
    load = _component(series, f"{key}_force_n", "z")
    sigma = _component(series, f"{key}_force_sigma_n", "z")
    if load is None or sigma is None:
        return False
    ax.fill_between(
        series.time_s,
        load - sigma,
        load + sigma,
        color=color,
        alpha=_BAND_ALPHA,
        linewidth=0,
        label=f"{label} {SIGMA_BAND_SUFFIX}",
    )
    return True


def _draw_measured_load(
    ax: Axes, series: GroundReactionPlotSeries, key: str, style: tuple[str, str]
) -> bool:
    """Dashed measured load and its band; True when the band was drawn."""
    label, color = style
    load = _component(series, f"{key}_force_n", "z")
    if load is None:
        return False
    ax.plot(series.time_s, load, color=color, label=label, **_MEASURED_STYLE)
    return _draw_band(ax, series, key, label, color)


def _draw_measured(
    axes: np.ndarray, series: GroundReactionPlotSeries, in_bw: bool
) -> tuple[bool, bool]:
    """Dashed measured load with its 1-sigma band (N panels only) and CoP path
    with its 1-sigma ellipses.

    Returns whether any load band and any CoP ellipse was drawn.
    """
    banded = ringed = False
    for key, label, color in _measured_keys(series):
        if not in_bw:
            banded = _draw_measured_load(axes[0], series, key, (label, color)) or banded
        x = _component(series, f"{key}_cop_m", "x")
        y = _component(series, f"{key}_cop_m", "y")
        if x is not None and y is not None:
            axes[3].plot(x, y, color=color, label=label, **_MEASURED_STYLE)
            ringed = _draw_cop_ellipses(axes[3], series, key, (label, color)) or ringed
    for ax in (axes[0], axes[3]):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="upper left")
    return banded, ringed


def _uncertainty_caption(
    series: GroundReactionPlotSeries, banded: bool, ringed: bool
) -> str | None:
    """What the band and ellipses show and where their uncertainty came from."""
    parts = []
    if banded:
        parts.append(f"shaded: measured load {SIGMA_BAND_SUFFIX}")
    if ringed:
        parts.append(f"ellipses: measured CoP {SIGMA_BAND_SUFFIX} {_PER_AXIS}")
    if not parts:
        return None
    text = "; ".join(parts)
    source = series.measured_uncertainty_source or "not stated"
    return f"{text[0].upper()}{text[1:]} (declared uncertainty: {source})"


def _draw_banner(
    fig: Figure, series: GroundReactionPlotSeries, caption: str | None
) -> None:
    lines = []
    if series.measured_qualified is False:
        lines.append((UNQUALIFIED_BANNER, "#B45309"))
    if caption is not None:
        lines.append((caption, "#4B5563"))
    for row, (text, color) in enumerate(lines):
        fig.text(0.01, 0.99 - 0.03 * row, text, color=color, fontsize=8, va="top")


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
    banded, ringed = _draw_measured(axes, series, in_bw)
    for ax in axes[4:]:
        ax.set_xlabel("Time (s)", fontsize=8)
    fig.suptitle("Ground Reaction (on the body, world axes)", fontsize=9)
    _draw_banner(fig, series, _uncertainty_caption(series, banded, ringed))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
