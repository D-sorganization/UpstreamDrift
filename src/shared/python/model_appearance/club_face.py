"""Clubface angle definitions shared by every engine (OSV-8, #11755).

Native world: +Z up, the target axis is -Y. The face angle is the horizontal
angle of the face normal relative to the target axis, positive when open for a
right-handed golfer (the face points to the golfer's right of the target
line, which is native -X), the GCV-15 definition.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

TARGET_AXIS = np.array([0.0, -1.0, 0.0])
MIN_HORIZONTAL_NORM = 1e-6


def horizontal_face_angle_deg(normal_world: Sequence[float] | np.ndarray) -> float:
    """Open-positive horizontal face angle of a world face normal, degrees.

    Raises ``ValueError`` for a non-finite, non-3-vector or vertical normal
    (no horizontal direction to measure). Postcondition: result in (-180, 180].
    """
    n = np.asarray(normal_world, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all():
        raise ValueError("normal_world must be a finite 3-vector")
    if math.hypot(n[0], n[1]) < MIN_HORIZONTAL_NORM:
        raise ValueError("the face normal has no horizontal component")
    return float(math.degrees(math.atan2(-n[0], -n[1])))


IMPACT_HEIGHT_TOL_M = 0.10
IMPACT_BALL_RADIUS_M = 0.15
_ACTIVE_FRACTION = 0.10


def _speeds(time: np.ndarray, clubhead: np.ndarray) -> np.ndarray:
    return (
        np.r_[0.0, np.linalg.norm(np.diff(clubhead, axis=0), axis=1)]
        / np.r_[1.0, np.diff(time)]
    )


def top_of_backswing_index(time: np.ndarray, clubhead: np.ndarray) -> int:
    """Transition frame: slowest clubhead sample between the first motion and peak speed."""
    from src.shared.python.motion_matching.loaders._align import detect_impact_index

    peak = detect_impact_index(time, clubhead)
    speed = _speeds(time, clubhead)
    start = int(np.argmax(speed > _ACTIVE_FRACTION * speed[peak]))
    if start >= peak:
        raise ValueError("no backswing precedes the peak clubhead speed")
    return start + int(np.argmin(speed[start:peak]))


def impact_frame(
    time: Sequence[float] | np.ndarray,
    clubhead: np.ndarray,
    *,
    height_tol_m: float = IMPACT_HEIGHT_TOL_M,
    ball_radius_m: float = IMPACT_BALL_RADIUS_M,
) -> int:
    """Impact index from the shared peak-speed detector, with sanity checks.

    ``clubhead`` is the ``(n, 3)`` face-centre world trajectory (Z up), whose
    first sample is the address position (the ball is there). The shared
    ``detect_impact_index`` (peak clubhead speed) is tried over the whole
    swing; when that frame is not at the ball (a matched swing peaks before the
    head arrives), the frame of closest approach to the address position after
    the top is used, provided it is within ``ball_radius_m`` of address.

    Raises ``ValueError`` when no frame satisfies: impact after the top of the
    backswing and clubhead height within ``height_tol_m`` of address. Never
    returns an unchecked frame.
    """
    from src.shared.python.motion_matching.loaders._align import detect_impact_index

    t = np.asarray(time, dtype=float)
    head = np.asarray(clubhead, dtype=float)
    if head.ndim != 2 or head.shape[1] != 3 or t.shape != (len(head),):
        raise ValueError("time must be (n,) and clubhead (n, 3)")
    if height_tol_m <= 0.0 or ball_radius_m <= 0.0:
        raise ValueError("tolerances must be positive")
    top = top_of_backswing_index(t, head)
    address = head[0]

    def ok(i: int) -> bool:
        return i > top and abs(head[i, 2] - address[2]) <= height_tol_m

    whole = detect_impact_index(t, head)
    if ok(whole):
        return int(whole)
    # The face meets the ball where it returns closest to its address position.
    after = np.arange(top + 1, len(head))
    idx = int(after[np.argmin(np.linalg.norm(head[after] - address, axis=1))])
    if ok(idx) and np.linalg.norm(head[idx] - address) <= ball_radius_m:
        return idx
    raise ValueError(
        "no valid impact frame: peak-speed frame "
        f"{whole} is not within {height_tol_m} m of the address height after the "
        f"top of the backswing (frame {top}), and the head never returns to the ball"
    )


def ball_passage(
    time: Sequence[float] | np.ndarray,
    clubhead: np.ndarray,
    *,
    height_tol_m: float = IMPACT_HEIGHT_TOL_M,
    ball_radius_m: float = IMPACT_BALL_RADIUS_M,
) -> tuple[float, int, float]:
    """Sub-sample impact: where the face centre passes the ball (OSV-10).

    At 45 m/s the head moves 5-12 cm per sample (1 kHz to 360 Hz) while the
    face turns 5-7 degrees, so a frame index is too coarse to compare face
    angles at impact. The face-centre path is taken piecewise linear; among
    the segments after the top of the backswing, the one passing closest to
    the address position (the ball) is impact, and the closest point on it
    gives the fraction ``s`` in ``[0, 1]``.

    Returns ``(t_impact, k, s)`` with ``t_impact = t[k] + s (t[k+1] - t[k])``.
    Raises ``ValueError`` for bad shapes or tolerances, or when no segment
    after the top comes within ``ball_radius_m`` of the ball with its closest
    point within ``height_tol_m`` of the address height.
    """
    t = np.asarray(time, dtype=float)
    head = np.asarray(clubhead, dtype=float)
    if head.ndim != 2 or head.shape[1] != 3 or t.shape != (len(head),):
        raise ValueError("time must be (n,) and clubhead (n, 3)")
    if not (np.isfinite(t).all() and np.isfinite(head).all()):
        raise ValueError("time and clubhead must be finite")
    if height_tol_m <= 0.0 or ball_radius_m <= 0.0:
        raise ValueError("tolerances must be positive")
    top = top_of_backswing_index(t, head)
    ball = head[0]
    start, step = head[top:-1], np.diff(head[top:], axis=0)
    length2 = np.einsum("ij,ij->i", step, step)
    s = np.clip(
        np.einsum("ij,ij->i", ball - start, step) / np.maximum(length2, 1e-18),
        0.0,
        1.0,
    )
    closest = start + s[:, None] * step
    gap = np.linalg.norm(closest - ball, axis=1)
    gap[np.abs(closest[:, 2] - ball[2]) > height_tol_m] = np.inf
    j = int(np.argmin(gap))
    if not gap[j] <= ball_radius_m:
        raise ValueError(
            f"no valid impact: the face centre never passes within {ball_radius_m} m "
            f"of the ball after the top of the backswing (frame {top})"
        )
    k = top + j
    return float(t[k] + s[j] * (t[k + 1] - t[k])), k, float(s[j])


def face_angle_at(normals: np.ndarray, k: int, s: float) -> float:
    """Open-positive face angle between samples ``k`` and ``k + 1``.

    The world face normals ``(n, 3)`` are blended linearly at fraction ``s``
    and renormalised (exact for the small inter-sample rotations). Raises
    ``ValueError`` for an out-of-range index or fraction or a non-finite
    normal.
    """
    n = np.asarray(normals, dtype=float)
    if n.ndim != 2 or n.shape[1] != 3:
        raise ValueError("normals must be (n, 3)")
    if not 0 <= k < len(n) - 1 or not 0.0 <= s <= 1.0:
        raise ValueError("k must index a segment and s lie in [0, 1]")
    blend = (1.0 - s) * n[k] + s * n[k + 1]
    return horizontal_face_angle_deg(blend)
