"""Constant-marker-set preprocessing and branch-continuity checks (#13056).

Frame-by-frame IK jumps whenever the set of visible markers changes: dropping
one club trio for the other, or a marker that appears late, moves the fit
even when the data are smooth. :func:`constant_marker_set` removes those
switches before IK by dropping named mostly-missing markers, rebuilding a
missing rigid group from a visible one, spline-bridging short interior gaps
and trimming an incomplete tail. Every filled sample is reported, and an
interior gap that cannot be filled is an error, never silently skipped.

:func:`largest_coordinate_steps` ranks the frame-to-frame rotational
coordinate steps of an IK result, so continuity is measured, not assumed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from src.shared.python.motion_matching.tour_capture_contract import TourCapture

__all__ = [
    "MarkerSetPolicy",
    "constant_marker_set",
    "largest_coordinate_steps",
]


@dataclass(frozen=True)
class MarkerSetPolicy:
    """What :func:`constant_marker_set` may change.

    ``rigid_groups`` are marker groups on one rigid body (three or more
    markers each); a fully missing group is rebuilt from another fully
    visible one. ``dropped`` markers are removed. Gaps up to
    ``max_spline_gap`` frames are bridged by a cubic spline.
    """

    rigid_groups: tuple[tuple[str, ...], ...] = ()
    dropped: tuple[str, ...] = ()
    max_spline_gap: int = 10

    def __post_init__(self) -> None:
        if any(len(group) < 3 for group in self.rigid_groups):
            raise ValueError("each rigid group needs at least three markers")
        if self.max_spline_gap < 1:
            raise ValueError("max_spline_gap must be at least 1 frame")

    def labels(self) -> set[str]:
        """Every label this policy names."""
        return {m for group in self.rigid_groups for m in group} | set(self.dropped)


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation taking the ``src`` points onto ``dst``."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cd - r @ cs


def _rigid_fill(
    points: np.ndarray, valid: np.ndarray, groups: list[list[int]]
) -> np.ndarray:
    """Rebuild each fully missing group from a fully visible one, in place.

    The rigid relation is taken from the nearest frame where both groups are
    visible. Returns the filled mask.
    """
    filled = np.zeros_like(valid)
    for missing in groups:
        for present in groups:
            if present is missing:
                continue
            both = np.flatnonzero(
                valid[:, missing].all(axis=1) & valid[:, present].all(axis=1)
            )
            rows = np.flatnonzero(
                ~valid[:, missing].any(axis=1) & valid[:, present].all(axis=1)
            )
            if both.size == 0:
                continue
            for f in rows:
                ref = both[np.argmin(np.abs(both - f))]
                r, t = _kabsch(points[ref, present], points[f, present])
                points[f, missing] = points[ref, missing] @ r.T + t
                valid[f, missing] = True
                filled[f, missing] = True
    return filled


def _spline_fill(points: np.ndarray, valid: np.ndarray, max_gap: int) -> np.ndarray:
    """Cubic-spline interior gaps up to ``max_gap`` frames, in place."""
    filled = np.zeros_like(valid)
    frames = np.arange(points.shape[0])
    for j in range(points.shape[1]):
        ok = valid[:, j]
        if ok.all() or ok.sum() < 4:
            continue
        spline = CubicSpline(frames[ok], points[ok, j])
        first, last = frames[ok][0], frames[ok][-1]
        gap = np.flatnonzero(~ok)
        for run in np.split(gap, np.flatnonzero(np.diff(gap) > 1) + 1):
            if run[0] > first and run[-1] < last and run.size <= max_gap:
                points[run, j] = spline(run)
                valid[run, j] = True
                filled[run, j] = True
    return filled


def _filled_rows(mask: np.ndarray, labels: Sequence[str]) -> dict[str, list[int]]:
    return {
        labels[j]: np.flatnonzero(mask[:, j]).tolist()
        for j in range(len(labels))
        if mask[:, j].any()
    }


def constant_marker_set(
    capture: TourCapture, policy: MarkerSetPolicy
) -> tuple[TourCapture, dict[str, Any]]:
    """Return ``capture`` with one marker set visible on every frame.

    Postconditions: every returned marker is valid on every returned frame;
    the report lists the dropped markers, every rigid- and spline-filled
    frame (indices into the returned capture) and the trimmed tail length.

    Raises:
        TypeError: ``capture`` or ``policy`` of the wrong type.
        ValueError: a policy label missing from the capture, no complete
            frame, or an interior gap that no fill could close.
    """
    if not isinstance(capture, TourCapture):
        raise TypeError("capture must be a TourCapture")
    if not isinstance(policy, MarkerSetPolicy):
        raise TypeError("policy must be a MarkerSetPolicy")
    unknown = sorted(policy.labels() - set(capture.labels))
    if unknown:
        raise ValueError(f"policy markers not in the capture: {unknown}")
    keep = [i for i, m in enumerate(capture.labels) if m not in policy.dropped]
    labels = [capture.labels[i] for i in keep]
    points = capture.points_m[:, keep].copy()
    valid = capture.valid[:, keep].copy()
    groups = [[labels.index(m) for m in g] for g in policy.rigid_groups]
    rigid = _rigid_fill(points, valid, groups)
    spline = _spline_fill(points, valid, policy.max_spline_gap)
    complete = valid.all(axis=1)
    if not complete.any():
        raise ValueError("no frame has every kept marker after filling")
    end = int(np.flatnonzero(complete)[-1]) + 1
    holes = np.flatnonzero(~complete[:end])
    if holes.size:
        raise ValueError(f"interior marker gaps remain at frames {holes.tolist()}")
    out = TourCapture(capture.time_s[:end], tuple(labels), points[:end], valid[:end])
    report = {
        "dropped_markers": list(policy.dropped),
        "rigid_filled": _filled_rows(rigid[:end], labels),
        "spline_filled": _filled_rows(spline[:end], labels),
        "trimmed_trailing_frames": int(capture.frames - end),
        "max_spline_gap_frames": policy.max_spline_gap,
    }
    return out, report


def largest_coordinate_steps(
    q: np.ndarray,
    names: Sequence[str],
    rotational: Sequence[bool],
    *,
    count: int = 5,
) -> list[dict[str, Any]]:
    """The ``count`` largest frame-to-frame rotational steps, largest first.

    Each entry is ``{"frame", "coordinate", "step_rad"}``, where ``frame`` is
    the later frame of the step. Translational coordinates are ignored.

    Raises:
        ValueError: ``q`` is not ``(frames, coordinates)`` matching ``names``
            and ``rotational``.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != len(names) or len(rotational) != len(names):
        raise ValueError(
            "q must be (frames, coordinates) matching names and rotational"
        )
    rot = np.flatnonzero(np.asarray(rotational, dtype=bool))
    if q.shape[0] < 2 or rot.size == 0:
        return []
    step = np.abs(np.diff(q[:, rot], axis=0))
    worst = step.max(axis=1)
    order = np.argsort(-worst, kind="stable")[:count]
    return [
        {
            "frame": int(k + 1),
            "coordinate": names[rot[step[k].argmax()]],
            "step_rad": float(worst[k]),
        }
        for k in order
    ]
