"""Shared helpers for drawing simple 3D skeletons in matplotlib.

The motion-matching diagnostics produce overlays of two poses (specified
vs. actual). These helpers keep the drawing primitives in one place so
each diagnostic doesn't reinvent the same matplotlib boilerplate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
from matplotlib.axes import Axes

from ..body_skeleton import BodySegment, default_body_segments

_DEFAULT_BODY_COLOR_MAP: dict[str, str] = {
    "torso": "#444444",
    "head": "#666666",
    "left_arm": "#1f77b4",
    "right_arm": "#ff7f0e",
    "left_leg": "#2ca02c",
    "right_leg": "#d62728",
    "pelvis": "#9467bd",
}


class _BodyTargetLike(Protocol):
    """Structural protocol for the subset of ``BodyTarget`` the renderer needs."""

    marker_xyz: np.ndarray  # (N, M, 3)
    marker_names: tuple[str, ...]


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


def draw_body_target_frame(
    ax: Axes,
    target: _BodyTargetLike,
    frame_idx: int,
    *,
    segment_groups: Sequence[str] | None = None,
    color_map: dict[str, str] | None = None,
    linewidth: float = 1.5,
) -> None:
    """Render one frame of a full-body marker target as a stick figure.

    Draws each segment from :func:`default_body_segments` (filtered to
    markers present on ``target``) as a coloured line on ``ax``. Segments
    whose endpoints contain any NaN coordinate at this frame are silently
    skipped, so missing/occluded markers degrade gracefully.

    Parameters
    ----------
    ax
        A matplotlib 3D axis (``projection='3d'``).
    target
        Anything exposing ``marker_xyz`` (shape ``(N, M, 3)``, metres) and
        ``marker_names`` (length-``M`` tuple of strings) — typically a
        ``BodyTarget`` instance.
    frame_idx
        Frame index into ``target.marker_xyz``.
    segment_groups
        Optional iterable of group names; only segments in these groups
        are drawn. ``None`` (default) draws every group.
    color_map
        Optional override mapping ``group -> matplotlib colour``. Missing
        keys fall back to the default colour map.
    linewidth
        Stroke width for every drawn segment.
    """
    xyz = np.asarray(target.marker_xyz)
    names = tuple(target.marker_names)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(
            f"target.marker_xyz must have shape (N, M, 3); got {xyz.shape!r}"
        )
    if xyz.shape[1] != len(names):
        raise ValueError(
            "target.marker_names length must equal marker_xyz.shape[1]; "
            f"got {len(names)} vs {xyz.shape[1]}"
        )
    n_frames = xyz.shape[0]
    if not (0 <= frame_idx < n_frames):
        raise ValueError(f"frame_idx {frame_idx} out of range [0, {n_frames})")

    groups_filter: frozenset[str] | None = (
        frozenset(segment_groups) if segment_groups is not None else None
    )
    colours = dict(_DEFAULT_BODY_COLOR_MAP)
    if color_map is not None:
        colours.update(color_map)

    name_to_idx = {n: i for i, n in enumerate(names)}
    segments: tuple[BodySegment, ...] = default_body_segments(names)
    frame = xyz[frame_idx]

    for seg in segments:
        if groups_filter is not None and seg.group not in groups_filter:
            continue
        ia = name_to_idx[seg.a]
        ib = name_to_idx[seg.b]
        pa = frame[ia]
        pb = frame[ib]
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            [pa[2], pb[2]],
            color=colours.get(seg.group, "#888888"),
            linewidth=linewidth,
        )
