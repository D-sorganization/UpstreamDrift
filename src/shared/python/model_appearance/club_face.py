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
