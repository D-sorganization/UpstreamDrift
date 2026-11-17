"""Signed-distance primitives and polynomial smooth-min for the SDF hand.

Distances are metres. Negative values are inside the solid. The grip is the
infinite cylinder ``x**2 + y**2 = R**2`` in the hand-grip frame, so overlap
with the hand is an analytic min of the two interiors — no mesh required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class Primitive:
    """One SDF atom in the hand-grip frame.

    Cones run from ``start`` (radius ``radius0``) to ``end`` (radius
    ``radius1``). Spheres use ``start`` as the centre and ``radius0``.
    Ellipsoids use ``start`` as the centre, ``end`` as the along-axis
    direction, ``palmar`` as the palmar axis, and ``radii`` as
    ``(side, palmar, along)`` radii. ``aspect`` scales the palmar
    cross-section of a cone (1 keeps a circular section).
    """

    name: str
    group: str
    kind: str
    operation: str
    shape: str
    start: Vector3
    end: Vector3
    radius0: float
    radius1: float
    palmar: Vector3
    aspect: float
    blend: str
    radii: Vector3


def smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth-min (Quilez). ``k`` is the blend radius in metres.

    Postcondition: where ``|a-b| >= k`` the result equals ``min(a, b)``.
    At ``a == b`` the result is ``a - k/6``.
    """
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError(f"blend radius k must be positive and finite, got {k!r}")
    gap = np.abs(a - b)
    h = np.maximum(k - gap, 0.0) / k
    return np.minimum(a, b) - (h * h * h) * k * (1.0 / 6.0)


def smax(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth-max, the dual of :func:`smin`."""
    return -smin(-a, -b, k)


def ssub(base: np.ndarray, cutter: np.ndarray, k: float) -> np.ndarray:
    """Smooth subtraction of ``cutter`` from ``base`` (crease grooves)."""
    return smax(base, -cutter, k)


def sd_round_cone(
    local: np.ndarray, radius0: float, radius1: float, length: float
) -> np.ndarray:
    """Round cone along local +y from y=0 to y=``length``.

    ``local`` is ``(N, 3)`` with x = side and z = palmar. Caps are spheres,
    so the solid extends ``radius0`` before the start and ``radius1`` past
    the end. Preconditions: positive length and radii, and a radius change
    that does not exceed the length (the cone slope stays real).
    """
    if length <= 0.0 or radius0 <= 0.0 or radius1 <= 0.0:
        raise ValueError("round cone needs a positive length and radii")
    slope = (radius0 - radius1) / length
    slope = float(np.clip(slope, -0.99, 0.99))
    lateral = math.sqrt(1.0 - slope * slope)
    qx = np.hypot(local[:, 0], local[:, 2])
    qy = local[:, 1]
    side = qx * (-slope) + qy * lateral
    cap0 = side < 0.0
    cap1 = side > lateral * length
    mid = ~(cap0 | cap1)
    out = np.empty(len(local), dtype=float)
    out[cap0] = np.hypot(qx[cap0], qy[cap0]) - radius0
    out[cap1] = np.hypot(qx[cap1], qy[cap1] - length) - radius1
    out[mid] = qx[mid] * lateral + qy[mid] * slope - radius0
    return out


def _orthogonal(direction: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Unit ``direction`` with its component along ``axis`` removed."""
    dropped = direction - float(np.dot(direction, axis)) * axis
    norm = float(np.linalg.norm(dropped))
    if norm >= 1e-8:
        return dropped / norm
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, axis))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    crossed = np.cross(axis, helper)
    return crossed / np.linalg.norm(crossed)


def cone_sdf(
    points: np.ndarray,
    start: Vector3,
    end: Vector3,
    radius0: float,
    radius1: float,
    palmar: Vector3,
    aspect: float,
) -> np.ndarray:
    """Tapered round-cone SDF. ``aspect`` < 1 flattens the palmar section."""
    if not 0.2 <= aspect <= 2.0:
        raise ValueError(f"cone aspect must be in [0.2, 2], got {aspect!r}")
    origin = np.asarray(start, dtype=float)
    axis = np.asarray(end, dtype=float) - origin
    length = float(np.linalg.norm(axis))
    if length < 1e-8:
        return np.full(len(points), np.inf)
    axis_u = axis / length
    pal = _orthogonal(np.asarray(palmar, dtype=float), axis_u)
    side = np.cross(axis_u, pal)
    rel = points - origin
    local = np.column_stack((rel @ side, rel @ axis_u, (rel @ pal) / aspect))
    return sd_round_cone(local, radius0, radius1, length) * min(aspect, 1.0)


def sphere_sdf(points: np.ndarray, centre: Vector3, radius: float) -> np.ndarray:
    """Exact sphere SDF."""
    if radius <= 0.0:
        raise ValueError(f"sphere radius must be positive, got {radius!r}")
    return np.linalg.norm(points - np.asarray(centre, dtype=float), axis=1) - radius


def ellipsoid_sdf(
    points: np.ndarray,
    centre: Vector3,
    along: Vector3,
    palmar: Vector3,
    radii: Vector3,
) -> np.ndarray:
    """Approximate ellipsoid SDF. ``radii`` is ``(side, palmar, along)``."""
    radii_a = np.asarray(radii, dtype=float)
    if np.any(radii_a <= 0.0):
        raise ValueError(f"ellipsoid radii must be positive, got {radii!r}")
    axis = np.asarray(along, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return np.full(len(points), np.inf)
    axis_u = axis / norm
    pal = _orthogonal(np.asarray(palmar, dtype=float), axis_u)
    side = np.cross(axis_u, pal)
    local = (points - np.asarray(centre, dtype=float)) @ np.column_stack(
        (side, pal, axis_u)
    )
    k0 = np.linalg.norm(local / radii_a, axis=1)
    k1 = np.linalg.norm(local / (radii_a * radii_a), axis=1)
    sd = k0 * (k0 - 1.0) / np.maximum(k1, 1e-8)
    return np.where(k0 < 1e-8, -float(np.min(radii_a)), sd)


def primitive_sdf(primitive: Primitive, points: np.ndarray) -> np.ndarray:
    """SDF of one primitive. Unknown shapes raise ``ValueError``."""
    if primitive.shape == "cone":
        return cone_sdf(
            points,
            primitive.start,
            primitive.end,
            primitive.radius0,
            primitive.radius1,
            primitive.palmar,
            primitive.aspect,
        )
    if primitive.shape == "sphere":
        return sphere_sdf(points, primitive.start, primitive.radius0)
    if primitive.shape == "ellipsoid":
        return ellipsoid_sdf(
            points, primitive.start, primitive.end, primitive.palmar, primitive.radii
        )
    raise ValueError(f"unknown SDF shape {primitive.shape!r}")


def _union_group(
    primitives: list[Primitive],
    points: np.ndarray,
    blends: dict[str, float],
) -> np.ndarray:
    field = np.full(len(points), np.inf)
    nails: list[Primitive] = []
    for primitive in primitives:
        if primitive.blend == "nail":
            nails.append(primitive)
            continue
        field = smin(field, primitive_sdf(primitive, points), blends[primitive.blend])
    for primitive in nails:
        field = smin(field, primitive_sdf(primitive, points), blends[primitive.blend])
    return field


def evaluate_hand_sdf(
    primitives: tuple[Primitive, ...],
    points: np.ndarray,
    blends: dict[str, float],
) -> np.ndarray:
    """Hand SDF at ``points`` of shape ``(N, 3)``.

    Digits are a hard min of per-finger smooth unions so neighbouring fingers
    stay separate. Each digit then smooth-mins into the palm. The thumb
    smooth-mins with the web blend so the first web fills. Creases are a
    smooth subtraction.

    Preconditions: ``points`` is a finite ``(N, 3)`` array and every blend
    key used by a primitive is present. Postcondition: one finite distance
    per point.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("points must be finite")
    grouped: dict[str, list[Primitive]] = {}
    creases: list[Primitive] = []
    for primitive in primitives:
        if primitive.blend not in blends and primitive.operation == "union":
            raise ValueError(
                f"primitive {primitive.name!r} blend {primitive.blend!r} is unset"
            )
        if primitive.operation == "subtract":
            creases.append(primitive)
        else:
            grouped.setdefault(primitive.group, []).append(primitive)
    digits = np.full(len(pts), np.inf)
    for name in ("index", "middle", "ring", "little"):
        if name in grouped:
            digits = np.minimum(digits, _union_group(grouped[name], pts, blends))
    palm = _union_group(grouped.get("palm", []), pts, blends)
    hand = smin(digits, palm, blends["mcp"])
    thumb = _union_group(grouped.get("thumb", []), pts, blends)
    hand = smin(hand, thumb, blends["web"])
    for crease in creases:
        if crease.blend not in blends:
            raise ValueError(f"crease blend {crease.blend!r} is unset")
        hand = ssub(hand, primitive_sdf(crease, pts), blends[crease.blend])
    return hand


def analytic_grip_penetration(
    sdf_values: np.ndarray,
    points: np.ndarray,
    grip_radius_m: float,
) -> np.ndarray:
    """Overlap depth (m) of the hand volume with the grip cylinder.

    Positive only where the point is inside both solids. The depth is the
    minimum of the hand interior (``-sdf``) and the cylinder interior
    (``R - hypot(x, y)``). Zero means no penetration at that sample.

    Preconditions: ``grip_radius_m > 0`` and ``sdf_values`` aligns with
    ``points``. Postcondition: a non-negative depth per point.
    """
    if not math.isfinite(grip_radius_m) or grip_radius_m <= 0.0:
        raise ValueError(f"grip_radius_m must be positive, got {grip_radius_m!r}")
    sdf = np.asarray(sdf_values, dtype=float)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or sdf.shape != (len(pts),):
        raise ValueError("sdf_values must be one finite distance per (N, 3) point")
    inside_grip = grip_radius_m - np.hypot(pts[:, 0], pts[:, 1])
    overlap = np.minimum(inside_grip, -sdf)
    return np.maximum(overlap, 0.0)


__all__ = [
    "Primitive",
    "analytic_grip_penetration",
    "cone_sdf",
    "ellipsoid_sdf",
    "evaluate_hand_sdf",
    "primitive_sdf",
    "sd_round_cone",
    "smax",
    "smin",
    "sphere_sdf",
    "ssub",
]
