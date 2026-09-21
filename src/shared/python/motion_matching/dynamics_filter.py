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

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]
GRAVITY_M_S2 = 9.81


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
    lengths = np.linalg.norm(normals, axis=1)
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
