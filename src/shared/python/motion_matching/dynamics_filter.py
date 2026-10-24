"""Dynamics filter for a tracked reference (MM-7b, #10109).

A marker-matched reference is kinematically faithful but not dynamically
consistent for the model that will replay it: its zero-moment point (the
point on the ground where the required reaction would act without moment)
can leave the support polygon, and no unilateral foot contact can realise
such frames. This module holds the model-free part of the correction:

* ``project_inside`` moves a zero-moment point into the support polygon
  shrunk by a safety margin (the nearest such point);
* ``cart_table_shift`` finds the smallest, smoothest centre-of-mass shift
  whose cart-table effect ``dz = dc - (z_c / g) dc''`` closes the gap between
  the reference zero-moment point and its projected target over the whole
  trajectory (Kajita's cart-table model: the moment of a point mass at
  height ``z_c`` moves the zero-moment point by ``-(z_c / g)`` times its
  horizontal acceleration).

The kinematic side (re-solving the pose against markers with the shifted
centre-of-mass targets) lives in the marker solver; the whole-body
zero-moment point comes from the simulator. Nothing here enters the
equations of motion: the filter only changes the reference the controller
tracks.
"""

from __future__ import annotations
from src.shared.python.core.physics_constants import GRAVITY_M_S2

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]
GRAVITY_M_S2 = float(GRAVITY_M_S2)


def _hull_edges(hull_xy: Array) -> tuple[Array, Array]:
    """Outward unit normals and offsets of a counter-clockwise convex hull."""
    pts = np.asarray(hull_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("A hull needs at least three 2-D vertices")
    if not np.isfinite(pts).all():
        raise ValueError("Hull vertices must be finite")
    centroid = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0]))
    pts = pts[order]
    edges = np.roll(pts, -1, axis=0) - pts
    normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)  # outward for CCW
    lengths = np.sqrt(np.einsum("ij,ij->i", normals, normals))
    if (lengths < 1e-12).any():
        raise ValueError("Hull has a degenerate edge")
    normals /= lengths[:, None]
    offsets = np.einsum("ij,ij->i", normals, pts)
    return normals, offsets


def project_inside(point_xy: Array, hull_xy: Array, margin_m: float = 0.0) -> Array:
    """Nearest point to ``point_xy`` inside the hull shrunk by ``margin_m``.

    Cyclic projection onto the violated half-planes (Dykstra for convex
    sets); a point already inside the shrunk hull is returned unchanged.
    Precondition: the margin leaves the shrunk hull non-empty (every vertex
    farther than ``margin_m`` from the centroid); the centroid is returned
    when it does not.
    """
    p = np.asarray(point_xy, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError("point_xy must be a finite 2-vector")
    if margin_m < 0:
        raise ValueError("margin must be nonnegative")
    normals, offsets = _hull_edges(hull_xy)
    offsets = offsets - margin_m
    centroid = np.asarray(hull_xy, dtype=float).mean(axis=0)
    if (normals @ centroid - offsets > 0).any():
        return centroid
    x = p.copy()
    for _ in range(200):
        violation = normals @ x - offsets
        worst = int(np.argmax(violation))
        if violation[worst] <= 1e-12:
            break
        x = x - violation[worst] * normals[worst]
    return x


def cart_table_shift(
    zmp_xy: Array,
    target_xy: Array,
    z_c_m: Array | float,
    times: Sequence[float] | Array,
    *,
    acceleration_weight: float = 1e-3,
    position_weight: float = 1e-2,
    pinned_frames: int = 2,
) -> Array:
    """Centre-of-mass shift ``dc`` (frames, 2) whose cart-table effect closes
    the gap ``target - zmp``.

    Solves, per axis, the least-squares problem
    ``min ||dc - (z_c / g) D2 dc - gap||^2 + wa ||D2 dc||^2 + wp ||dc||^2``
    with the first ``pinned_frames`` rows held at zero (the address is
    already consistent); ``D2`` is the second finite difference. The
    regularisers keep the shift small and smooth so it moves the
    zero-moment point mostly through position, not through acceleration
    bursts. Postcondition: shape (frames, 2), zero on the pinned frames.
    """
    zmp = np.asarray(zmp_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    t = np.asarray(times, dtype=float)
    n = t.size
    if zmp.shape != (n, 2) or target.shape != (n, 2) or n < 3:
        raise ValueError("zmp and target must be (frames, 2) over at least three times")
    if (
        np.any(np.diff(t) <= 0)
        or not np.isfinite(zmp).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("times must increase and inputs be finite")
    height = np.broadcast_to(np.asarray(z_c_m, dtype=float), (n,))
    if (height <= 0).any():
        raise ValueError("centre-of-mass height must be positive")
    if min(acceleration_weight, position_weight) < 0 or pinned_frames < 0:
        raise ValueError("weights and pinned frames must be nonnegative")
    dt = float(np.mean(np.diff(t)))
    d2 = np.zeros((n, n))
    for k in range(1, n - 1):
        d2[k, k - 1 : k + 2] = [1.0, -2.0, 1.0]
    d2 /= dt * dt
    operator = np.eye(n) - (height / GRAVITY_M_S2)[:, None] * d2
    gap = target - zmp
    rows = np.vstack(
        [
            operator,
            np.sqrt(acceleration_weight) * d2 * dt * dt,
            np.sqrt(position_weight) * np.eye(n),
        ]
    )
    pin = min(pinned_frames, n)
    keep = np.arange(pin, n)
    shift = np.zeros((n, 2))
    for axis in range(2):
        rhs = np.concatenate([gap[:, axis], np.zeros(2 * n)])
        solution, *_ = np.linalg.lstsq(rows[:, keep], rhs, rcond=None)
        shift[keep, axis] = solution
    return shift


def _convex_hull(points_xy: Array) -> Array:
    """Counter-clockwise convex hull of 2-D points (Andrew's monotone chain)."""
    pts = np.unique(np.asarray(points_xy, dtype=float), axis=0)
    if pts.shape[0] < 3:
        return pts

    def cross(o: Array, a: Array, b: Array) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[Array] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Array] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def support_min_width(points_xy: Array) -> float:
    """Narrowest width of the support polygon spanned by ``points_xy`` (m).

    The minimum over directions of the polygon's extent, attained across one
    of its convex-hull edges (rotating-calipers argument). For planted feet
    this is the anteroposterior depth of the base of support, about one foot
    length; on a single foot it is about the foot's width. Fewer than three
    distinct points (a line or a point) have zero width.
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        raise ValueError("points_xy must be finite (points, 2)")
    hull = _convex_hull(pts)
    if hull.shape[0] < 3:
        return 0.0
    edges = np.roll(hull, -1, axis=0) - hull
    normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
    normals /= np.sqrt(np.einsum("ij,ij->i", normals, normals))[:, None]
    extent = normals @ hull.T  # (edges, vertices)
    return float((extent.max(axis=1) - extent.min(axis=1)).min())


def bound_com_shift(shift: Array, limit_m: Array | float) -> tuple[Array, Array]:
    """Radially clip each frame's centre-of-mass shift to ``limit_m`` (#13112).

    The cart-table correction is a re-posing of a marker-matched reference,
    not a force: it moves the quasi-static zero-moment point by the shift
    itself (``dz = dc`` when ``dc''`` vanishes). With planted feet the
    whole-body centre of mass must stay over the base of support, so no
    admissible correction carries it farther than the base's narrowest width
    (one foot length in double stance): a larger shift would move a centred
    centre of mass clean across the base. A gap larger than that is not a
    centre-of-mass position error but an acceleration the reference demands
    (typically a frame whose vertical reaction is barely above the unloaded
    threshold, so its zero-moment point lies metres outside the feet), and
    re-posing the body cannot fix it. The direction is kept; only the
    magnitude is limited.

    Returns:
        (bounded shift (frames, 2), boolean mask of the clipped frames).
    """
    s = np.asarray(shift, dtype=float)
    if s.ndim != 2 or s.shape[1] != 2 or not np.isfinite(s).all():
        raise ValueError("shift must be finite (frames, 2)")
    limit = np.broadcast_to(np.asarray(limit_m, dtype=float), (s.shape[0],))
    if (limit < 0).any() or not np.isfinite(limit).all():
        raise ValueError("limit must be finite and nonnegative")
    norm = np.sqrt(np.einsum("ij,ij->i", s, s))
    clipped = norm > limit
    scale = np.ones_like(norm)
    scale[clipped] = limit[clipped] / norm[clipped]
    return s * scale[:, None], clipped
