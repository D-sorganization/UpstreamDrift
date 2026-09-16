"""Ground support contracts shared by every full-body engine lane.

Three engine-agnostic pieces the ground-supported full-body work needs once:
the capture-to-native world mapping (the tour capture is Y-up, the native
model world is Z-up with ``(x, y, z) -> (x, -z, y)``, reused from the marker
loaders), the ground-height calibration from the lowest toe markers, and the
static support report (total normal force against body weight, centre of
pressure, support-polygon containment) evaluated from shared contact samples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.contact_law import ContactSample, GroundPlane
from src.shared.python.motion_matching.loaders._marker_clusters import (
    y_up_to_z_up_rotation,
)

Array: TypeAlias = NDArray[np.float64]


def capture_to_native_world(points_y_up: Array) -> Array:
    """Map Y-up capture points to the native Z-up world, ``(x, y, z) -> (x, -z, y)``.

    NaN entries (missing samples) pass through. Precondition: last axis is 3.
    Postcondition: the mapping is a proper rotation, so distances are kept.
    """
    points = np.asarray(points_y_up, dtype=float)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("Capture points must have a trailing axis of length 3")
    rotation = y_up_to_z_up_rotation()
    # Index mapping rather than a matmul so NaN samples do not spread across axes.
    out = np.empty_like(points)
    for row in range(3):
        column = int(np.flatnonzero(rotation[row])[0])
        out[..., row] = rotation[row, column] * points[..., column]
    return out


@dataclass(frozen=True)
class GroundCalibration:
    """Ground height derived from the lowest valid marker of the chosen labels."""

    height_m: float
    lowest_marker_height_m: float
    standoff_m: float
    labels: tuple[str, ...]
    frames: int
    policy: str = "lowest valid toe marker minus standoff, along the ground normal"


def calibrate_ground_height(
    points_native: Array,
    valid: NDArray[Any],
    labels: Sequence[str],
    toe_labels: Sequence[str],
    *,
    normal: Sequence[float] = (0.0, 0.0, 1.0),
    standoff_m: float,
) -> GroundCalibration:
    """Set the ground plane height from the lowest valid toe marker.

    Preconditions: ``points_native`` is (frames, markers, 3) with ``valid`` of
    shape (frames, markers); every toe label is a capture label with at least
    one valid sample; ``standoff_m`` (marker centre to sole) is finite and
    nonnegative. Postcondition: ``height_m`` equals the lowest toe-marker
    height along ``normal`` minus the standoff.
    """
    points = np.asarray(points_native, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3 or mask.shape != points.shape[:2]:
        raise ValueError("Points must be (frames, markers, 3) with matching validity")
    if len(labels) != points.shape[1] or len(set(labels)) != len(labels):
        raise ValueError("Labels must be unique and match the marker axis")
    if not toe_labels or any(label not in labels for label in toe_labels):
        raise ValueError("Toe labels must be a nonempty subset of the capture labels")
    if not np.isfinite(standoff_m) or standoff_m < 0:
        raise ValueError("Standoff must be finite and nonnegative")
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
        raise ValueError("Ground normal must be a finite nonzero 3-vector")
    n = n / np.linalg.norm(n)
    columns = [list(labels).index(label) for label in toe_labels]
    heights = points[:, columns, :] @ n
    usable = mask[:, columns] & np.isfinite(heights)
    if not usable.any():
        raise ValueError("No valid toe marker sample to calibrate the ground")
    lowest = float(np.min(heights[usable]))
    return GroundCalibration(
        height_m=lowest - float(standoff_m),
        lowest_marker_height_m=lowest,
        standoff_m=float(standoff_m),
        labels=tuple(toe_labels),
        frames=int(points.shape[0]),
    )


def _plane_basis(normal: Array) -> tuple[Array, Array]:
    helper = (
        np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    u = np.asarray(np.cross(normal, helper), dtype=np.float64)
    u /= np.linalg.norm(u)
    return u, np.asarray(np.cross(normal, u), dtype=np.float64)


def convex_hull_contains(
    point_xy: Array, points_xy: Array, tolerance_m: float = 0.0
) -> bool:
    """True when ``point_xy`` lies inside (or within ``tolerance_m`` of) the hull.

    Precondition: at least three finite 2-D points spanning an area; two points
    or collinear sets have no interior and return False.
    """
    pts = np.unique(np.asarray(points_xy, dtype=float), axis=0)
    p = np.asarray(point_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or p.shape != (2,):
        raise ValueError("Hull test needs (N, 2) points and a 2-vector")
    if not np.isfinite(pts).all() or not np.isfinite(p).all():
        raise ValueError("Hull test needs finite points")
    if pts.shape[0] < 3:
        return False
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o: Array, a: Array, b: Array) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[Array] = []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list[Array] = []
    for q in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return False
    for i, a in enumerate(hull):
        b = hull[(i + 1) % len(hull)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if cross(a, b, p) / length < -tolerance_m:
            return False
    return True


@dataclass(frozen=True)
class SupportReport:
    """Static support evaluation of one contact state."""

    total_normal_force_n: float
    weight_n: float
    weight_fraction: float
    centre_of_pressure_m: tuple[float, float, float] | None
    inside_support_polygon: bool
    active_spheres: tuple[str, ...]


def support_report(
    samples: Mapping[str, ContactSample],
    sphere_ground_points: Mapping[str, Array],
    plane: GroundPlane,
    mass_kg: float,
    gravity_m_s2: Sequence[float],
) -> SupportReport:
    """Summarise how the contact spheres carry the body at one state.

    ``sphere_ground_points`` are the sphere centres projected onto the plane
    (the support polygon vertices, including spheres not in contact).
    Precondition: positive mass, finite gravity, every sample has a polygon
    point. Postcondition: ``weight_fraction`` is total normal force divided by
    ``mass * |g|``; the centre of pressure is the normal-force-weighted mean
    of the contact points, None when nothing touches.
    """
    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("Mass must be finite and positive")
    g = np.asarray(gravity_m_s2, dtype=float)
    if g.shape != (3,) or not np.isfinite(g).all():
        raise ValueError("Gravity must be a finite 3-vector")
    missing = [name for name in samples if name not in sphere_ground_points]
    if missing:
        raise ValueError(f"Samples without polygon points: {missing}")
    n = np.asarray(plane.normal, dtype=float)
    n = n / np.linalg.norm(n)
    weight = float(mass_kg * np.linalg.norm(g))
    total = 0.0
    weighted = np.zeros(3)
    active: list[str] = []
    for name, sample in samples.items():
        magnitude = float(np.asarray(sample.normal_force_n) @ n)
        if magnitude <= 0.0:
            continue
        active.append(name)
        total += magnitude
        weighted += magnitude * np.asarray(sample.contact_point_m, dtype=float)
    u, v = _plane_basis(n)
    polygon = np.array([[p @ u, p @ v] for p in sphere_ground_points.values()])
    if total > 0.0:
        cop = weighted / total
        inside = convex_hull_contains(np.array([cop @ u, cop @ v]), polygon)
        cop_out: tuple[float, float, float] | None = (
            float(cop[0]),
            float(cop[1]),
            float(cop[2]),
        )
    else:
        cop_out, inside = None, False
    return SupportReport(
        total_normal_force_n=total,
        weight_n=weight,
        weight_fraction=total / weight,
        centre_of_pressure_m=cop_out,
        inside_support_polygon=inside,
        active_spheres=tuple(active),
    )
