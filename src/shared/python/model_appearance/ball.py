"""Golf ball placement at address, shared by every consumer (OSV-3, GCV-13).

There is exactly one definition of where the ball sits at address: the
face-centre contact point offset by one ball radius along the face normal,
resting on the ground. Gaze targets (``motion_matching.gaze``) and ball
rendering must both call :func:`ball_position_at_address`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

# Regulation ball diameter 42.67 mm (R&A / USGA minimum), radius 21.335 mm.
BALL_RADIUS_M: float = 0.021335


def ball_position_at_address(
    face_centre_m: Sequence[float] | NDArray[np.float64],
    face_normal: Sequence[float] | NDArray[np.float64],
    *,
    ground_height_m: float = 0.0,
    radius_m: float = BALL_RADIUS_M,
) -> NDArray[np.float64]:
    """World position of the ball centre at address.

    The centre is the face-centre contact point offset by ``radius_m`` along
    the unit face normal, then lowered/raised so that it rests on the ground
    (z = ``ground_height_m + radius_m``; Z is up). The horizontal position is
    the face-normal offset of the contact point; a lofted normal's vertical
    component is discarded because a ball on the ground cannot sit higher.

    Preconditions: ``face_centre_m`` and ``face_normal`` are finite 3-vectors,
    the normal is nonzero, ``radius_m`` is positive.
    Postcondition: returned z equals ``ground_height_m + radius_m``.
    """
    centre = np.asarray(face_centre_m, dtype=float)
    normal = np.asarray(face_normal, dtype=float)
    if centre.shape != (3,) or normal.shape != (3,):
        raise ValueError("face centre and face normal must be 3-vectors")
    if not (np.isfinite(centre).all() and np.isfinite(normal).all()):
        raise ValueError("face centre and face normal must be finite")
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        raise ValueError("face normal must be nonzero")
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be positive")
    ball = centre + radius_m * normal / length
    ball[2] = ground_height_m + radius_m
    return ball
