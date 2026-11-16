"""Engine-free capsule geometry helpers (HRG-GB2)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _finite_vec3(val: np.ndarray, name: str) -> np.ndarray:
    if (
        not isinstance(val, np.ndarray)
        or val.shape != (3,)
        or not np.isfinite(val).all()
    ):
        raise ValueError(f"{name} must be a finite 3-vector")
    return val.astype(np.float64, copy=False)


def closest_points_segments(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest points on two finite segments."""
    a0 = _finite_vec3(a0, "a0")
    a1 = _finite_vec3(a1, "a1")
    b0 = _finite_vec3(b0, "b0")
    b1 = _finite_vec3(b1, "b1")
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    eps = 1e-14
    if a < eps and e < eps:
        return a0.copy(), b0.copy()
    if a < eps:
        t = float(np.clip(-(r @ d2) / e, 0.0, 1.0))
        return a0.copy(), b0 + t * d2
    if e < eps:
        s = float(np.clip(-(r @ d1) / a, 0.0, 1.0))
        return a0 + s * d1, b0.copy()
    b = float(d1 @ d2)
    c = float(d1 @ r)
    f = float(d2 @ r)
    denom = max(a * e - b * b, 0.0)
    if denom < 1e-10 * a * e:
        s = 0.5
        t = float(np.clip((b * s + f) / e, 0.0, 1.0))
    else:
        s = (b * f - c * e) / denom
        t = (b * s + f) / e
        s = float(np.clip(s, 0.0, 1.0))
        t = float(np.clip(t, 0.0, 1.0))
    return a0 + s * d1, b0 + t * d2


def capsule_capsule_signed_distance(
    p0_a: np.ndarray,
    p1_a: np.ndarray,
    radius_a: float,
    p0_b: np.ndarray,
    p1_b: np.ndarray,
    radius_b: float,
) -> float:
    """Signed clearance between two capsule solids (negative = penetration)."""
    pa, pb = closest_points_segments(p0_a, p1_a, p0_b, p1_b)
    dist = float(np.linalg.norm(pb - pa))
    return dist - radius_a - radius_b


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix for a unit *axis* and *angle_rad*."""
    axis = _finite_vec3(axis, "axis")
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("axis must be non-zero")
    k = axis / norm
    x, y, z = float(k[0]), float(k[1]), float(k[2])
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def deg_to_rad(angle_deg: float) -> float:
    if not math.isfinite(angle_deg):
        raise ValueError("angle_deg must be finite")
    return math.radians(angle_deg)
