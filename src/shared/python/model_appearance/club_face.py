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
