"""Conditioning of matched-IK coordinate trajectories before kinetics (#11739).

Why.  Joint-wise marker IK can flip a solution branch for a single frame (or
for the rest of the trajectory, by a multiple of 2 pi).  Kinematics tolerate
that, but kinetics differentiate the motion twice, so one flipped frame turns
into a kN-scale impulse on the club.  The driver OpenSim IK candidate has 15
such frames (steps of up to 3 rad in one 2.8 ms frame, against a median of
0.001 rad), e.g. at 0.947 s and 1.258 s; filtering alone only spreads them.

Method (documented, lossless where possible):

1. :func:`unwrap_angular` removes 2 pi branch flips of every angular
   coordinate (translations are not wrapped).
2. :func:`detect_ik_outliers` flags a frame when at least ``min_coordinates``
   angular coordinates deviate from their running median (window of
   ``2*half_window + 1`` frames) by more than ``threshold_rad``.  A few
   coordinates deviating is ordinary motion; many deviating at once is an IK
   failure.
3. :func:`condition_trajectory` replaces flagged frames, all coordinates, by
   a shape-preserving cubic (PCHIP) interpolation across the valid frames
   and returns a report of what was replaced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter

TRANSLATION_PREFIX = "Translation"


@dataclass(frozen=True)
class ConditioningReport:
    """What :func:`condition_trajectory` changed."""

    repaired_frames: int
    frame_times_s: list[float]
    unwrapped_coordinates: list[str]


def _angular_mask(names: Sequence[str]) -> np.ndarray:
    return np.array([not n.startswith(TRANSLATION_PREFIX) for n in names])


def _check(q: np.ndarray, names: Sequence[str]) -> np.ndarray:
    arr = np.asarray(q, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError(f"q must be (n, {len(names)}) to match names, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("q must be finite")
    return arr


def unwrap_angular(q: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Remove 2 pi branch flips of the angular columns of ``q``."""
    arr = _check(q, names).copy()
    mask = _angular_mask(names)
    arr[:, mask] = np.unwrap(arr[:, mask], axis=0)
    return arr


def detect_ik_outliers(
    q: np.ndarray,
    names: Sequence[str],
    *,
    threshold_rad: float = 0.35,
    min_coordinates: int = 2,
    half_window: int = 5,
) -> np.ndarray:
    """Boolean mask of frames where many angular coordinates jump together."""
    arr = _check(q, names)
    if threshold_rad <= 0.0 or min_coordinates < 1 or half_window < 1:
        raise ValueError("threshold, min_coordinates and half_window must be positive")
    ang = arr[:, _angular_mask(names)]
    median = median_filter(ang, size=(2 * half_window + 1, 1), mode="nearest")
    return np.asarray(
        np.sum(np.abs(ang - median) > threshold_rad, axis=1) >= min_coordinates
    )


def first_discontinuity_time(
    time_s: np.ndarray,
    q: np.ndarray,
    names: Sequence[str],
    *,
    jump_rad: float = 0.5,
    min_coordinates: int = 3,
) -> float | None:
    """Time of the first frame where many angular coordinates jump at once.

    A persistent solution-branch switch of the IK (several joints stepping by
    more than ``jump_rad`` within one frame, against a typical per-frame
    change of 1e-3 rad) cannot be repaired by interpolation, because the
    poses before and after it are different IK solutions.  Kinetics are only
    defined before the first such frame.  Returns ``None`` when there is none.
    Angles are unwrapped first, so a pure 2 pi flip does not count.
    """
    t = np.asarray(time_s, dtype=float)
    arr = unwrap_angular(q, names)
    if t.ndim != 1 or t.size != arr.shape[0]:
        raise ValueError("time_s must match the rows of q")
    jumps = np.abs(np.diff(arr[:, _angular_mask(names)], axis=0)) > jump_rad
    hit = np.flatnonzero(jumps.sum(axis=1) >= min_coordinates)
    return float(t[hit[0] + 1]) if hit.size else None


def condition_trajectory(
    time_s: np.ndarray,
    q: np.ndarray,
    names: Sequence[str],
    **detect_kwargs: float,
) -> tuple[np.ndarray, ConditioningReport]:
    """Unwrap angles, then interpolate over IK-failure frames.

    Raises:
        ValueError: for a non-increasing or too short time base, a name/column
            mismatch, or when too few valid frames remain to interpolate.
    """
    t = np.asarray(time_s, dtype=float)
    arr = unwrap_angular(q, names)
    if t.ndim != 1 or t.size != arr.shape[0] or t.size < 4 or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be strictly increasing and match q rows (>= 4)")
    bad = detect_ik_outliers(arr, names, **detect_kwargs)  # type: ignore[arg-type]
    good = ~bad
    if good.sum() < 4:
        raise ValueError("fewer than 4 valid frames remain after outlier detection")
    out = arr.copy()
    if bad.any():
        out[bad] = PchipInterpolator(t[good], arr[good], axis=0)(t[bad])
    unwrapped = [
        n
        for n, m, c in zip(names, _angular_mask(names), range(len(names)), strict=True)
        if m and not np.allclose(arr[:, c], np.asarray(q, dtype=float)[:, c])
    ]
    return out, ConditioningReport(
        int(bad.sum()), [float(x) for x in t[bad]], unwrapped
    )
