"""One time-series panel shared by the force plot renderers (GCV-5, #11711).

Unavailable samples are NaN and leave gaps; a panel with no available sample
says "unavailable" with the reason instead of drawing a flat zero line.  Event
markers (address, top, impact, finish) are dashed vertical lines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from matplotlib.axes import Axes
import numpy as np

__all__ = [
    "EVENT_COLOR",
    "UNAVAILABLE_COLOR",
    "draw_event_markers",
    "draw_time_panel",
    "mark_unavailable",
]

EVENT_COLOR = "#F87171"
UNAVAILABLE_COLOR = "#B45309"


def draw_event_markers(ax: Axes, events: Mapping[str, float]) -> None:
    """Dashed vertical line and label at every named event time."""
    for name, when in events.items():
        ax.axvline(when, color=EVENT_COLOR, linestyle="--", linewidth=1.0)
        ax.annotate(
            name,
            (when, 1.0),
            xycoords=("data", "axes fraction"),
            fontsize=7,
            color=EVENT_COLOR,
            va="top",
            ha="right",
        )


def mark_unavailable(ax: Axes, reason: str) -> None:
    """Centre an "unavailable: <reason>" note on ``ax``."""
    ax.text(
        0.5,
        0.5,
        f"unavailable: {reason or 'not provided by this engine'}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color=UNAVAILABLE_COLOR,
    )


def draw_time_panel(
    ax: Axes,
    time_s: Sequence[float],
    rows: Sequence[tuple[np.ndarray, str, str]],
    *,
    title: str,
    unit: str,
    events: Mapping[str, float],
    reason: str = "",
) -> None:
    """Plot ``rows`` of ``(values, label, colour)`` against ``time_s``.

    Postcondition: the panel shows a legend when it has several rows with data
    and an "unavailable" note when no row has a finite sample.

    Raises:
        ValueError: if a row's length differs from ``time_s``.
    """
    t = np.asarray(time_s, dtype=float)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(unit, fontsize=8)
    ax.grid(True, alpha=0.3)
    shown = 0
    for values, label, color in rows:
        y = np.asarray(values, dtype=float)
        if y.shape != t.shape:
            raise ValueError(
                f"row {label!r} length {y.size} must equal time_s length {t.size}"
            )
        shown += int(np.isfinite(y).any())
        ax.plot(t, y, color=color, linewidth=1.6, label=label)
    draw_event_markers(ax, events)
    if shown == 0:
        mark_unavailable(ax, reason)
    elif len(rows) > 1:
        ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)
