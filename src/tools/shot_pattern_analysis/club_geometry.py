"""Rigid static-loft, forward-lean and shaft-twist geometry for matched studies.

This module deliberately differs from the historical constant-delivered-loft
sensitivity. Positive lean pitches the entire club toward the target. Positive
shaft twist closes a right-handed face. Angles are degrees; vectors use the
native right-handed frame (+X target, +Y golfer/left, +Z up).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClubDelivery:
    """Orthonormal face frame and shaft axis after ordered rigid rotations."""

    normal: tuple[float, float, float]
    heel_axis: tuple[float, float, float]
    up_axis: tuple[float, float, float]
    shaft_axis: tuple[float, float, float]
    face_deg: float
    loft_deg: float


def _tuple3(vector: np.ndarray) -> tuple[float, float, float]:
    return float(vector[0]), float(vector[1]), float(vector[2])


def club_delivery(
    static_loft_deg: float,
    lie_deg: float,
    lean_deg: float,
    nominal_face_deg: float,
    shaft_rotation_deg: float,
) -> ClubDelivery:
    """Pitch club, yaw nominal setup, then rotate about its delivered shaft.

    Zero twist gives loft = static loft minus lean. Yaw is an actual input
    orientation, never an endpoint rotation. The shaft/face angle is invariant.
    ``heel_axis`` points toward the right-handed golfer; toe is its negative.
    """
    values = (static_loft_deg, lie_deg, lean_deg, nominal_face_deg, shaft_rotation_deg)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all club geometry angles must be finite")
    if not 0 < static_loft_deg < 80 or not 0 < lie_deg < 90:
        raise ValueError("static loft must be in (0,80), lie in (0,90) degrees")
    if abs(lean_deg) > 30 or abs(nominal_face_deg) > 30 or abs(shaft_rotation_deg) > 30:
        raise ValueError(
            "lean, nominal face and shaft rotation must be within 30 degrees"
        )

    loft, lie, lean, yaw, twist = map(math.radians, values)
    # Face-frame columns are normal, heel, and face-up (a proper rotation).
    face = np.array(
        [
            [math.cos(loft), 0, -math.sin(loft)],
            [0, 1, 0],
            [math.sin(loft), 0, math.cos(loft)],
        ],
        dtype=float,
    )
    pitch = np.array(
        [
            [math.cos(lean), 0, math.sin(lean)],
            [0, 1, 0],
            [-math.sin(lean), 0, math.cos(lean)],
        ],
        dtype=float,
    )
    # Golf-positive right is negative native z-axis yaw.
    heading = np.array(
        [
            [math.cos(yaw), math.sin(yaw), 0],
            [-math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    orientation = heading @ pitch
    shaft = orientation @ np.array([0.0, math.cos(lie), math.sin(lie)])
    x, y, z = shaft
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rotation = (
        math.cos(twist) * np.eye(3)
        + math.sin(twist) * cross
        + (1 - math.cos(twist)) * np.outer(shaft, shaft)
    )
    frame = rotation @ orientation @ face
    normal = frame[:, 0]
    return ClubDelivery(
        normal=_tuple3(normal),
        heel_axis=_tuple3(frame[:, 1]),
        up_axis=_tuple3(frame[:, 2]),
        shaft_axis=_tuple3(shaft),
        face_deg=math.degrees(math.atan2(-normal[1], normal[0])),
        loft_deg=math.degrees(math.atan2(normal[2], math.hypot(*normal[:2]))),
    )
