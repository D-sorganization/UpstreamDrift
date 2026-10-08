"""Grip force and couple plots (GCV-10, #11716).

One matplotlib figure, shared by the PyQt widget and the video export: hand
force magnitudes, net force at the grip midpoint, the equivalent couple about
the midpoint (world or club frame) and the contact-force-moment vs applied-free
torque split, each with the event markers and the ``split_method`` in the
figure title.  Unavailable samples are NaN and leave gaps; a panel with no
available sample says "unavailable" with the reason instead of a flat zero line.
"""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from src.shared.python.biomechanics.grip_plot_model import GripPlotSeries

__all__ = ["COUPLE_FRAMES", "plot_grip_wrench"]

COUPLE_FRAMES: tuple[str, ...] = ("world", "club")

#: GRIP #56B4E9 is the net colour; hands are lighter/darker (ADR-0052).
_COLORS = {
    "left": "#9AD0F2",
    "right": "#1B6C99",
    "net": "#56B4E9",
    "couple": "#E69F00",
    "contact": "#009E73",
    "free": "#CC79A7",
}
_UNAVAILABLE_COLOR = "#B45309"


def _values(series: GripPlotSeries, trace: str) -> np.ndarray:
    raw = series.traces[trace]["magnitude"]
    return np.array([np.nan if v is None else v for v in raw], dtype=float)


def _panel(
    ax: Axes,
    series: GripPlotSeries,
    title: str,
    unit: str,
    rows: Sequence[tuple[str, str, str]],
) -> None:
    t = np.asarray(series.time_s, dtype=float)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(unit, fontsize=8)
    ax.grid(True, alpha=0.3)
    shown = 0
    for trace, label, color in rows:
        y = _values(series, trace)
        if np.isfinite(y).any():
            shown += 1
        ax.plot(t, y, color=color, linewidth=1.6, label=label)
    for name, when in series.events.items():
        ax.axvline(when, color="#F87171", linestyle="--", linewidth=1.0)
        ax.annotate(
            name,
            (when, 1.0),
            xycoords=("data", "axes fraction"),
            fontsize=7,
            color="#F87171",
            va="top",
            ha="right",
        )
    if shown == 0:
        ax.text(
            0.5,
            0.5,
            f"unavailable: {series.reason or 'not provided by this engine'}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=_UNAVAILABLE_COLOR,
        )
    elif len(rows) > 1:
        ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)


def plot_grip_wrench(
    fig: Figure, series: GripPlotSeries, *, couple_frame: str = "world"
) -> None:
    """Draw the four grip panels onto ``fig`` (cleared first).

    Args:
        couple_frame: ``"world"`` or ``"club"`` for the couple panel.

    Raises:
        ValueError: for an unknown ``couple_frame``.
    """
    if couple_frame not in COUPLE_FRAMES:
        raise ValueError(f"couple_frame must be one of {COUPLE_FRAMES}")
    fig.clear()
    axes = fig.subplots(4, 1, sharex=True)
    couple = "couple_nm" if couple_frame == "world" else "couple_local_nm"
    _panel(
        axes[0],
        series,
        "Hand Force Magnitude",
        "N",
        [
            ("left_force_n", "Left", _COLORS["left"]),
            ("right_force_n", "Right", _COLORS["right"]),
        ],
    )
    _panel(
        axes[1],
        series,
        "Net Force at Grip Midpoint",
        "N",
        [("net_force_n", "Net", _COLORS["net"])],
    )
    _panel(
        axes[2],
        series,
        f"Equivalent Couple at Midpoint ({couple_frame.capitalize()})",
        "N*m",
        [(couple, "Couple", _COLORS["couple"])],
    )
    _panel(
        axes[3],
        series,
        "Contact Force Moment vs Applied Free Torque",
        "N*m",
        [
            ("contact_force_moment_nm", "Contact force moment", _COLORS["contact"]),
            ("applied_free_torque_nm", "Free torque", _COLORS["free"]),
        ],
    )
    axes[3].set_xlabel("Time (s)", fontsize=8)
    fig.suptitle(
        f"Hands on Club (on the club, world axes) | split: {series.split_method}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
