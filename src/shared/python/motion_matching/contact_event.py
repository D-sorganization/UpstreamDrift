"""Bracketed sphere contact against a finite, moving triangle surface.

The caller supplies the production state-advance callback and the face mesh
transformed into world coordinates at each state. A root is accepted only for
a front-side, approaching passage through the sphere-to-surface signed gap;
the returned state is obtained by advancing the original state to the root.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

Array = NDArray[np.float64]
AdvanceFn = Callable[[float], tuple[Array, Array]]
FaceTrianglesFn = Callable[[Array], Array]
MISS_PENALTY_SUBDIVISIONS = 8
VertexKey = tuple[int, int, int]
EdgeKey = tuple[VertexKey, VertexKey]


@dataclass(frozen=True)
class ContactRoot:
    """Approaching sphere-to-face contact at an integrated state."""

    time_s: float
    q: Array
    qd: Array
    contact_point_m: Array
    contact_normal: Array
    approach_speed_mps: float
    triangle_index: int


@dataclass(frozen=True)
class ContactSearchResult:
    """Root, when found, and a finite smooth miss penalty in metres."""

    event: ContactRoot | None
    closest_approach_penalty_m: float


def _point_triangles(point: Array, triangles: Array) -> tuple[Array, Array]:
    """Vectorized closest points and barycentrics for the triangle set."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = point - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = point - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    count = len(triangles)
    closest = np.empty((count, 3), dtype=float)
    bary = np.empty((count, 3), dtype=float)
    assigned = np.zeros(count, dtype=bool)

    def set_rows(mask: np.ndarray, weights: Array) -> None:
        closest[mask] = np.einsum("ij,ijk->ik", weights[mask], triangles[mask])
        bary[mask] = weights[mask]
        assigned[mask] = True

    weights = np.zeros((count, 3))
    mask = (d1 <= 0.0) & (d2 <= 0.0)
    weights[mask, 0] = 1.0
    set_rows(mask, weights)
    mask = ~assigned & (d3 >= 0.0) & (d4 <= d3)
    weights[:] = 0.0
    weights[mask, 1] = 1.0
    set_rows(mask, weights)
    mask = ~assigned & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v = np.divide(d1, d1 - d3, out=np.zeros(count), where=(d1 - d3) != 0.0)
    weights[:] = 0.0
    weights[:, 0], weights[:, 1] = 1.0 - v, v
    set_rows(mask, weights)
    mask = ~assigned & (d6 >= 0.0) & (d5 <= d6)
    weights[:] = 0.0
    weights[mask, 2] = 1.0
    set_rows(mask, weights)
    mask = ~assigned & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w = np.divide(d2, d2 - d6, out=np.zeros(count), where=(d2 - d6) != 0.0)
    weights[:] = 0.0
    weights[:, 0], weights[:, 2] = 1.0 - w, w
    set_rows(mask, weights)
    mask = ~assigned & (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0)
    w = np.divide(
        d4 - d3,
        (d4 - d3) + (d5 - d6),
        out=np.zeros(count),
        where=((d4 - d3) + (d5 - d6)) != 0.0,
    )
    weights[:] = 0.0
    weights[:, 1], weights[:, 2] = 1.0 - w, w
    set_rows(mask, weights)
    mask = ~assigned
    denom = va + vb + vc
    if np.any(denom[mask] == 0.0):
        raise ValueError("face triangles must be nondegenerate")
    v = vb[mask] / denom[mask]
    w = vc[mask] / denom[mask]
    weights[mask] = np.column_stack((1.0 - v - w, v, w))
    set_rows(mask, weights)
    return closest, bary


def _surface_sample(
    triangles: Array, centre: Array, radius: float
) -> tuple[float, float, int, Array, Array, Array]:
    if len(triangles) == 0:
        raise ValueError("face_triangles_world must contain at least one triangle")
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(crosses, axis=1)
    if np.any(lengths <= np.finfo(float).tiny):
        raise ValueError("face triangles must be nondegenerate")
    closest, barycentrics = _point_triangles(centre, triangles)
    deltas = centre - closest
    distances2 = np.einsum("ij,ij->i", deltas, deltas)
    index = int(np.argmin(distances2))
    point, bary, delta = closest[index], barycentrics[index], deltas[index]
    cross = np.cross(
        triangles[index, 1] - triangles[index, 0],
        triangles[index, 2] - triangles[index, 0],
    )
    outward = cross / lengths[index]
    distance = math.sqrt(float(distances2[index]))
    side = float(delta @ outward)
    signed_distance = math.copysign(distance, side) if side != 0.0 else -distance
    gap = signed_distance - radius

    # A normalized soft minimum of squared distances stays positive and smooth
    # across triangle seams; radius supplies its physical length scale.
    scaled = -distances2 / (radius * radius)
    weights = np.exp(scaled - float(np.max(scaled)))
    weights /= float(weights.sum())
    smooth_distance = math.sqrt(float(weights @ distances2))
    penalty = radius * float(np.logaddexp(0.0, (smooth_distance - radius) / radius))
    return gap, penalty, index, point, bary, outward


def _boundary_edge_contact(triangles: Array, index: int, bary: Array) -> bool:
    """Whether a closest point lies on the outer edge of the triangle patch."""
    scale = max(1.0, float(np.max(np.abs(triangles))))
    quantum = 128.0 * np.finfo(float).eps * scale

    def key(point: Array) -> VertexKey:
        quantized = np.rint(point / quantum).astype(np.int64)
        return int(quantized[0]), int(quantized[1]), int(quantized[2])

    def edge_key(a: VertexKey, b: VertexKey) -> EdgeKey:
        return (a, b) if a <= b else (b, a)

    counts: Counter[EdgeKey] = Counter()
    for tri in triangles:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            counts[edge_key(key(tri[a]), key(tri[b]))] += 1
    tol = 128.0 * np.finfo(float).eps
    for opposite, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
        if bary[opposite] <= tol:
            edge = edge_key(key(triangles[index, a]), key(triangles[index, b]))
            if counts[edge] == 1:
                return True
    return False


def find_contact_event(
    advance: AdvanceFn,
    face_triangles_world: FaceTrianglesFn,
    ball_centre_m: Array,
    ball_radius_m: float,
    *,
    duration_s: float,
    time_tolerance_s: float = 1e-9,
    max_iterations: int = 80,
) -> ContactSearchResult:
    """Find a bracketed, front-side sphere contact during one actual advance.

    ``advance(t)`` must return ``(q, qd)`` by advancing the same initial state
    through the production integrator for ``t`` seconds. ``face_triangles_world(q)``
    returns the finite outward-wound surface triangles at that state. The ball
    is stationary in world coordinates. The signed gap is closest-point distance
    to the finite triangles (negative behind their outward normals), less the
    ball radius. Only a positive-to-nonpositive bracket is eligible.

    A root on the outer boundary of the supplied patch is rejected; shared
    interior triangle seams remain eligible. The caller must supply outward
    triangle winding. Only an endpoint sign-change bracket is searched, so a
    tangent touch or an even number of crossings in the interval is not found.
    The smooth closest-approach penalty is a soft minimum over nine equally
    spaced states. This bounded scan can miss a narrower or multimodal minimum;
    it is not a global closest-approach guarantee. The penalty is returned for
    hits and misses alike, so event acquisition does not switch its value.
    This finder does not select or interpolate generalized states.
    """
    centre = np.asarray(ball_centre_m, dtype=float)
    radius = float(ball_radius_m)
    duration = float(duration_s)
    tolerance = float(time_tolerance_s)
    if centre.shape != (3,) or not np.isfinite(centre).all():
        raise ValueError("ball_centre_m must be a finite three-vector")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("ball_radius_m must be positive and finite")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be positive and finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance >= duration:
        raise ValueError("time_tolerance_s must lie inside duration_s")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")

    def sample(
        time_s: float,
    ) -> tuple[Array, Array, Array, tuple[float, float, int, Array, Array, Array]]:
        q, qd = (np.asarray(x, dtype=float) for x in advance(time_s))
        if (
            q.ndim != 1
            or qd.ndim != 1
            or not np.isfinite(q).all()
            or not np.isfinite(qd).all()
        ):
            raise ValueError("advance must return finite one-dimensional q and qd")
        triangles = np.asarray(face_triangles_world(q), dtype=float)
        if (
            triangles.ndim != 3
            or triangles.shape[1:] != (3, 3)
            or not np.isfinite(triangles).all()
        ):
            raise ValueError(
                "face_triangles_world must return finite (F, 3, 3) triangles"
            )
        return q, qd, triangles, _surface_sample(triangles, centre, radius)

    lo, hi = 0.0, duration
    _, _, _, data_lo = sample(lo)
    _, _, _, data_hi = sample(hi)
    event: ContactRoot | None = None
    if data_lo[0] > 0.0 and data_hi[0] <= 0.0:
        for _ in range(max_iterations):
            if hi - lo <= tolerance:
                break
            mid = 0.5 * (lo + hi)
            _, _, _, data_mid = sample(mid)
            if data_mid[0] > 0.0:
                lo, data_lo = mid, data_mid
            else:
                hi, data_hi = mid, data_mid
        if hi - lo > tolerance:
            raise ValueError("contact root did not converge within max_iterations")
        time_s = 0.5 * (lo + hi)
        q_star, qd_star, triangles, data_star = sample(time_s)
        index, point, bary = data_star[2], data_star[3], data_star[4]
        delta = centre - point
        if not _boundary_edge_contact(triangles, index, bary):
            distance = float(np.linalg.norm(delta))
            if distance > 0.0:
                normal = delta / distance
                speed = float((data_lo[0] - data_hi[0]) / (hi - lo))
                if speed > 0.0:
                    event = ContactRoot(
                        time_s,
                        q_star.copy(),
                        qd_star.copy(),
                        point.copy(),
                        normal.copy(),
                        speed,
                        int(index),
                    )

    penalty_samples = np.asarray(
        [
            sample(duration * i / MISS_PENALTY_SUBDIVISIONS)[3][1]
            for i in range(MISS_PENALTY_SUBDIVISIONS + 1)
        ],
        dtype=float,
    )
    penalty = -radius * float(
        logsumexp(-penalty_samples / radius) - math.log(len(penalty_samples))
    )
    return ContactSearchResult(event, penalty)
