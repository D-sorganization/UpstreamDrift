"""Matplotlib drawing of an NSX :class:`PanelView` (NSX-43, #12575).

No PyQt and no physics: it draws the numbers and classes the view carries.
Bands are drawn first; a determined-sign segment is a solid fill, an
indeterminate one is shaded and an unbounded one is hatched.  A representative
is drawn only as an overlay on its band.  Artists carry a ``gid`` naming what
they are (``segment:<display class>``, ``overlay``, ``infeasible``) so the
tests can read back what was drawn.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from matplotlib.figure import Figure

from src.tools.grip_wrench_plots.nsx_view import CurveView, PanelView, SegmentView

__all__ = ["draw_panel"]

_STYLE: dict[str, dict[str, Any]] = {
    "DETERMINED_SIGN": {"facecolor": "#2a9d8f", "alpha": 0.55},
    "INDETERMINATE": {"facecolor": "#9aa0a6", "alpha": 0.35},
}
_UNBOUNDED_STYLE: dict[str, Any] = {
    "facecolor": "none",
    "edgecolor": "#c0392b",
    "hatch": "///",
}


def _nan(values: Sequence[float | None]) -> list[float]:
    return [math.nan if v is None else float(v) for v in values]


def _segment_span(curve: CurveView, seg: SegmentView) -> list[int]:
    return [
        k
        for k, frame in enumerate(curve.frames)
        if seg.first_frame_index <= frame <= seg.last_frame_index
    ]


def _draw_segment(ax: Any, curve: CurveView, seg: SegmentView) -> None:
    gid = f"segment:{seg.display_class}"
    if seg.display_class == "UNBOUNDED" or curve.lower is None or curve.upper is None:
        patch = ax.axvspan(
            seg.first_frame_index, seg.last_frame_index, **_UNBOUNDED_STYLE
        )
        patch.set_gid("segment:UNBOUNDED")
        return
    idx = _segment_span(curve, seg)
    x = [curve.frames[k] for k in idx]
    lower = _nan([curve.lower[k] for k in idx])
    upper = _nan([curve.upper[k] for k in idx])
    style = _STYLE[seg.display_class]
    if len(x) == 1:
        line = ax.vlines(x, lower, upper, linewidth=6, color=style["facecolor"])
        line.set_gid(gid)
        return
    fill = ax.fill_between(x, lower, upper, **style)
    fill.set_gid(gid)


def _draw_curve(ax: Any, curve: CurveView) -> None:
    ax.set_title(curve.name, fontsize=9, loc="left")
    ax.set_ylabel(curve.units, fontsize=8)
    for seg in curve.segments:
        _draw_segment(ax, curve, seg)
    for line in curve.lines:
        ax.plot(curve.frames, _nan(line.values), marker="o", label=line.label)
    if curve.overlay is not None:
        (art,) = ax.plot(
            curve.frames,
            _nan(curve.overlay.values),
            linestyle="--",
            color="black",
            marker="x",
            label="representative (overlay)",
        )
        art.set_gid("overlay")
    for frame in curve.infeasible_frames:
        ax.axvline(frame, color="#c0392b", linestyle=":").set_gid("infeasible")
    if curve.lines or curve.overlay is not None:
        ax.legend(fontsize=7, loc="best")


def draw_panel(figure: Figure, view: PanelView) -> None:
    """Redraw ``figure`` from ``view`` (one axes per curve).

    Postcondition: ``len(figure.axes) == len(view.curves)``; with no curve the
    figure holds one text line saying why.
    """
    figure.clear()
    if not view.curves:
        figure.text(
            0.5,
            0.5,
            view.status or "nothing to show",
            ha="center",
            va="center",
            wrap=True,
        )
        return
    axes = figure.subplots(len(view.curves), 1, squeeze=False)[:, 0]
    for ax, curve in zip(axes, view.curves, strict=True):
        _draw_curve(ax, curve)
    axes[-1].set_xlabel("frame index", fontsize=8)
    figure.tight_layout()
