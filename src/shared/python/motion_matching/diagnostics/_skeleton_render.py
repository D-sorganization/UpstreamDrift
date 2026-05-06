"""Shared helpers for drawing simple 3D skeletons in matplotlib.

The motion-matching diagnostics produce overlays of two poses (specified
vs. actual). These helpers keep the drawing primitives in one place so
each diagnostic doesn't reinvent the same matplotlib boilerplate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from matplotlib.axes import Axes


def draw_segments(
    ax: Axes,
    points: Sequence[np.ndarray],
    *,
    color: str,
    label: str | None = None,
    linewidth: float = 2.0,
    marker: str = "o",
) -> None:
    """Draw a single open polyline through ``points`` on a 3D axis.

    ``points`` is a sequence of ``(3,)`` arrays in metres. The first
    point is plotted with the supplied label so the legend stays clean
    even when called multiple times for paired skeletons.
    """
    if not points:
        return
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"draw_segments expects an Nx3 array of points, got shape {arr.shape}"
        )
    ax.plot(
        arr[:, 0],
        arr[:, 1],
        arr[:, 2],
        color=color,
        linewidth=linewidth,
        marker=marker,
        label=label,
    )


def draw_delta_arrows(
    ax: Axes,
    starts: Iterable[np.ndarray],
    ends: Iterable[np.ndarray],
    *,
    color: str = "black",
) -> int:
    """Draw arrows from each ``start`` to its paired ``end``.

    Returns the number of arrows drawn (handy for tests that want to
    assert "we actually rendered the deltas").
    """
    count = 0
    for s, e in zip(starts, ends, strict=True):
        s_arr = np.asarray(s, dtype=float).reshape(3)
        e_arr = np.asarray(e, dtype=float).reshape(3)
        delta = e_arr - s_arr
        if not np.any(delta):
            continue
        ax.quiver(
            s_arr[0],
            s_arr[1],
            s_arr[2],
            delta[0],
            delta[1],
            delta[2],
            color=color,
            arrow_length_ratio=0.15,
            linewidth=1.0,
        )
        count += 1
    return count


def equalize_3d_axes(ax: Axes, points: np.ndarray) -> None:
    """Set equal aspect ratio for a 3D axis around the bounding box of ``points``."""
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    span = float((maxs - mins).max())
    if span <= 0:
        span = 1.0
    half = 0.5 * span * 1.1
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)  # type: ignore[attr-defined]
