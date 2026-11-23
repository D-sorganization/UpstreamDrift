"""Bracketed sphere contact against a finite, moving triangle surface.

The caller supplies the production state-advance callback and the face mesh
transformed into world coordinates at each state. A root is accepted only for
a front-side, approaching passage through the sphere-to-surface signed gap;
the returned state is obtained by advancing the original state to the root.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

Array = NDArray[np.float64]
AdvanceFn = Callable[[float], tuple[Array, Array]]
FaceTrianglesFn = Callable[[Array], Array]
FaceVelocityFn = Callable[[Array, Array, Array], Array]
# Root bracketing is deliberately bounded; missed crossings between these
# states remain an explicit limitation.
CONTACT_ROOT_SUBDIVISIONS = 8
SIDE_SWITCH_REFINEMENT = 32
MISS_PENALTY_SUBDIVISIONS = 8
MIN_APPROACH_SPEED_MPS = 1e-7


@dataclass(frozen=True)
class _BoundaryTopology:
    """Boundary edge indices for one fixed, rigidly moving triangle mesh."""

    reference: Array
    boundary_edges: Array
    boundary_face_opposites: frozenset[tuple[int, int]]
    boundary_face_vertices: frozenset[tuple[int, int]]
    shared_vertex_pairs: Array

    def validate_rigid_pose(self, triangles: Array) -> None:
        if triangles.shape != self.reference.shape:
            raise ValueError("face topology changed during contact search")
        current = triangles.reshape(-1, 3)
        extent = float(np.max(np.ptp(current, axis=0)))
        scale = max(extent, np.finfo(float).tiny)
        world_magnitude = float(np.max(np.abs(current)))
        resolution = float(np.spacing(world_magnitude))
        if not math.isfinite(resolution) or resolution > 1e-8 * scale:
            raise ValueError("face geometry exceeds available coordinate resolution")
        if len(self.shared_vertex_pairs):
            differences = (
                current[self.shared_vertex_pairs[:, 0]]
                - current[self.shared_vertex_pairs[:, 1]]
            )
            if np.max(np.linalg.norm(differences, axis=1)) > max(
                1e-12 * scale, 2.0 * resolution
            ):
                raise ValueError("face topology changed during contact search")
        reference = self.reference.reshape(-1, 3)
        ref_mean, current_mean = reference.mean(axis=0), current.mean(axis=0)
        ref_centered, current_centered = reference - ref_mean, current - current_mean
        left, _, right = np.linalg.svd(ref_centered.T @ current_centered)
        rotation = left @ right
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        residual = current_centered - ref_centered @ rotation
        scale = max(1.0, float(np.linalg.norm(ref_centered, axis=1).max()))
        if (
            not np.isfinite(residual).all()
            or np.max(np.linalg.norm(residual, axis=1)) > 1e-10 * scale
        ):
            raise ValueError("face topology changed; rigid pose is required")


def _build_boundary_topology(triangles: Array) -> _BoundaryTopology:
    """Cache outer-edge indices while retaining the original quantized test."""
    vertices = triangles.reshape(-1, 3)
    scale = float(np.max(np.ptp(vertices, axis=0)))
    if not math.isfinite(scale) or scale <= np.finfo(float).tiny:
        raise ValueError("face geometry has insufficient topology resolution")
    world_magnitude = float(np.max(np.abs(vertices)))
    resolution = float(np.spacing(world_magnitude))
    if not math.isfinite(resolution) or resolution > 1e-8 * scale:
        raise ValueError("face geometry exceeds available coordinate resolution")
    quantum = 128.0 * np.finfo(float).eps * scale
    origin = vertices[0]
    edges = triangles[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2, 3)
    quantized = np.rint((edges - origin) / quantum).astype(np.int64)
    first, second = quantized[:, 0], quantized[:, 1]
    swap = (first[:, 0] > second[:, 0]) | (
        (first[:, 0] == second[:, 0])
        & (
            (first[:, 1] > second[:, 1])
            | ((first[:, 1] == second[:, 1]) & (first[:, 2] > second[:, 2]))
        )
    )
    keys = np.column_stack(
        (
            np.where(swap[:, None], second, first),
            np.where(swap[:, None], first, second),
        )
    )
    _, first_indices, edge_inverse, counts = np.unique(
        keys, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    vertex_keys = np.rint((vertices - origin) / quantum).astype(np.int64)
    _, vertex_inverse, vertex_counts = np.unique(
        vertex_keys, axis=0, return_inverse=True, return_counts=True
    )
    triangle_vertices = np.sort(vertex_inverse.reshape(-1, 3), axis=1)
    if len(np.unique(triangle_vertices, axis=0)) != len(triangles):
        raise ValueError("face topology contains duplicate triangles")
    if np.any(counts > 2):
        raise ValueError("face topology has a nonmanifold edge")
    directions = np.bincount(edge_inverse, weights=np.where(swap, -1, 1))
    if np.any((counts == 2) & (directions != 0)):
        raise ValueError("face winding is inconsistent across a shared edge")
    boundary_indices = first_indices[counts == 1]
    face_indices, local_edges = np.divmod(boundary_indices, 3)
    opposite_for_edge = (2, 0, 1)
    boundary_face_opposites = frozenset(
        (int(face), opposite_for_edge[int(edge)])
        for face, edge in zip(face_indices, local_edges, strict=True)
    )
    vertex_ids = vertex_inverse.reshape(-1, 3)
    edge_endpoints = np.array(((0, 1), (1, 2), (2, 0)))
    boundary_ids = np.unique(
        vertex_ids[face_indices[:, None], edge_endpoints[local_edges]]
    )
    boundary_face_vertices = frozenset(
        (int(face), int(vertex))
        for face, vertex in np.argwhere(np.isin(vertex_ids, boundary_ids))
    )
    shared_groups = [
        np.flatnonzero(vertex_inverse == index)
        for index in np.flatnonzero(vertex_counts > 1)
    ]
    shared_vertex_pairs = (
        np.concatenate(
            [
                np.column_stack((np.repeat(group[0], len(group) - 1), group[1:]))
                for group in shared_groups
            ],
            axis=0,
        )
        if shared_groups
        else np.empty((0, 2), dtype=np.int64)
    )
    return _BoundaryTopology(
        triangles.copy(),
        np.column_stack((face_indices, local_edges)),
        boundary_face_opposites,
        boundary_face_vertices,
        shared_vertex_pairs,
    )


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
    """Sampled candidate status and finite smooth proximity penalty.

    ``sampled_bracket_candidate`` is not a certification that this is the
    first contact; no sampled bracket remains unresolved between samples.
    """

    event: ContactRoot | None
    closest_approach_penalty_m: float
    coverage_status: str


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
    triangles: Array,
    centre: Array,
    radius: float,
    *,
    boundary_topology: _BoundaryTopology | None = None,
) -> tuple[float, float, int, Array, Array, Array]:
    if len(triangles) == 0:
        raise ValueError("face_triangles_world must contain at least one triangle")
    with np.errstate(over="ignore", invalid="ignore"):
        crosses = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
    lengths = np.linalg.norm(crosses, axis=1)
    if not np.isfinite(crosses).all() or not np.isfinite(lengths).all():
        raise ValueError("face geometry must produce finite distances and normals")
    if np.any(lengths <= np.finfo(float).tiny):
        raise ValueError("face triangles must be nondegenerate")
    closest, barycentrics = _point_triangles(centre, triangles)
    deltas = centre - closest
    with np.errstate(over="ignore", invalid="ignore"):
        distances2 = np.einsum("ij,ij->i", deltas, deltas)
    if not np.isfinite(distances2).all():
        raise ValueError("face geometry must produce finite distances and normals")
    index = int(np.argmin(distances2))
    point, bary, delta = closest[index], barycentrics[index], deltas[index]
    cross = np.cross(
        triangles[index, 1] - triangles[index, 0],
        triangles[index, 2] - triangles[index, 0],
    )
    outward = cross / lengths[index]
    # Smooth the normal-side sign numerically; negative signed distance means
    # the sphere centre is behind the outward-wound face.
    distance = math.sqrt(float(distances2[index]))
    side = float(delta @ outward)
    side_smoothing = math.sqrt(np.finfo(float).eps) * radius
    smooth_side = math.tanh(side / side_smoothing)
    signed_distance = distance * smooth_side
    gap = signed_distance - radius

    # A normalized soft minimum of squared distances and blended triangle
    # side values stay continuous at seams. This is a per-triangle surrogate:
    # uneven tessellation affects the weights, not the geometric root.
    scaled = -distances2 / (radius * radius)
    weights = np.exp(scaled - float(np.max(scaled)))
    weights /= float(weights.sum())
    smooth_distance = math.sqrt(float(weights @ distances2))
    triangle_sides = np.einsum("ij,ij->i", deltas, crosses / lengths[:, None])
    blended_side = float(weights @ np.tanh(triangle_sides / side_smoothing))
    smooth_gap = smooth_distance * blended_side - radius
    gap_smoothing = math.sqrt(np.finfo(float).eps) * radius
    gap_penalty = math.hypot(smooth_gap, gap_smoothing) - gap_smoothing

    # Penalize proximity to the outside boundary continuously. Edge lengths
    # weight the Gaussian proximity sum so subdividing a boundary does not
    # change its scale. Squared segment distance keeps the term finite and
    # smooth at segment endpoints.
    boundary_penalty = _boundary_proximity_penalty(
        triangles, centre, radius, topology=boundary_topology
    )
    penalty = gap_penalty + boundary_penalty
    if not math.isfinite(gap) or not math.isfinite(penalty):
        raise ValueError("contact gap and miss cost must be finite")
    return gap, penalty, index, point, bary, outward


def _boundary_proximity_penalty(
    triangles: Array,
    point: Array,
    radius: float,
    *,
    topology: _BoundaryTopology | None = None,
) -> float:
    """Finite smooth cost for proximity to the finite patch's outer boundary."""
    edges = triangles[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2, 3)
    if topology is None:
        topology = _build_boundary_topology(triangles)
    face_indices, local_edges = topology.boundary_edges.T
    boundary = edges[face_indices * 3 + local_edges]
    vectors = boundary[:, 1] - boundary[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    fractions = np.clip(
        np.einsum("ij,ij->i", point - boundary[:, 0], vectors)
        / np.einsum("ij,ij->i", vectors, vectors),
        0.0,
        1.0,
    )
    deltas = point - (boundary[:, 0] + fractions[:, None] * vectors)
    distances2 = np.einsum("ij,ij->i", deltas, deltas)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        proximity = float(
            np.sum((lengths / radius) * np.exp(-distances2 / (radius * radius)))
        )
        penalty = radius * math.sqrt(proximity)
    if not math.isfinite(penalty):
        raise ValueError("contact boundary cost must be finite")
    return penalty


def _boundary_edge_contact(
    triangles: Array,
    index: int,
    bary: Array,
    *,
    topology: _BoundaryTopology | None = None,
) -> bool:
    """Whether a closest point lies on the outer edge of the triangle patch."""
    if topology is None:
        topology = _build_boundary_topology(triangles)
    tol = 128.0 * np.finfo(float).eps
    if np.count_nonzero(bary <= tol) >= 2:
        if (index, int(np.argmax(bary))) in topology.boundary_face_vertices:
            return True
    for opposite in range(3):
        if bary[opposite] <= tol:
            if (index, opposite) in topology.boundary_face_opposites:
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
    distance_tolerance_m: float = 1e-6,
    max_iterations: int = 80,
    face_point_velocity_world: FaceVelocityFn | None = None,
) -> ContactSearchResult:
    """Find a bracketed, front-side sphere contact during one actual advance.

    ``advance(t)`` must return ``(q, qd)`` by advancing the same initial state
    through the production integrator for ``t`` seconds. ``face_triangles_world(q)``
    returns the finite outward-wound surface triangles at that state, in a
    fixed triangle order and connectivity, undergoing rigid motion only. Each
    returned mesh is checked against that contract; deformation, reordering,
    or changed topology fails closed. The ball is stationary in world
    coordinates. The sphere clearance is the unsigned closest-point distance
    to the finite triangles minus the ball radius. It is positive outside and
    negative during overlap, including across triangle seams. Only a
    positive-to-nonpositive bracket on this clearance is eligible; the root
    must separately lie on the outward-facing side of the selected triangle.
    A reported candidate must meet both the bracket time tolerance and the
    specified absolute sphere-clearance tolerance. The default one-micrometre
    clearance tolerance is a numerical root criterion, not a model-accuracy or
    physical-contact uncertainty bound.
    The optional velocity callback supplies the instantaneous world velocity
    of the face material point at ``(q, qd, point_world)``. A candidate also
    requires a positive face velocity toward the fixed ball; without that
    callback a sampled bracket remains unresolved. This instantaneous test is
    not a bound between samples or a guarantee of first contact. Approach at
    or below 1e-7 m/s is treated as numerically ambiguous, not physical proof.

    A root on the outer boundary of the supplied patch is rejected; shared
    interior triangle seams remain eligible. The caller must supply outward
    triangle winding. A fixed grid of nine equally spaced states is searched
    for a positive-to-nonpositive clearance bracket, which is then bisected.
    A coarse interval with a side-sign switch but no clearance bracket is
    additionally sampled at 32 subdivisions; a side switch alone is never a
    contact. A
    separating start can therefore be followed by a later sampled contact.
    Narrow crossings between grid states, tangent touches, and multiple
    crossings within one grid cell can still be missed or unresolved.
    The smooth closest-approach penalty is a soft minimum over nine equally
    spaced states. This bounded scan can miss a narrower or multimodal minimum;
    it is not a global closest-approach guarantee. The penalty is returned for
    hits and misses alike, so event acquisition does not switch its value.
    Its per-triangle soft weights depend on the mesh tessellation and it is
    not a tessellation-invariant distance measurement.
    This finder does not select or interpolate generalized states.

    Limitation: a point-only rigid fit cannot distinguish a planar reflection
    from a proper 3D rotation that flips the plane normal. The caller remains
    responsible for preserving the physical outward-winding convention.
    """
    centre = np.asarray(ball_centre_m, dtype=float)
    radius = float(ball_radius_m)
    duration = float(duration_s)
    tolerance = float(time_tolerance_s)
    distance_tolerance = float(distance_tolerance_m)
    if centre.shape != (3,) or not np.isfinite(centre).all():
        raise ValueError("ball_centre_m must be a finite three-vector")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("ball_radius_m must be positive and finite")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be positive and finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance >= duration:
        raise ValueError("time_tolerance_s must lie inside duration_s")
    if not math.isfinite(distance_tolerance) or distance_tolerance <= 0.0:
        raise ValueError("distance_tolerance_m must be positive and finite")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")

    boundary_topology: _BoundaryTopology | None = None

    def sample(
        time_s: float,
    ) -> tuple[Array, Array, Array, tuple[float, float, int, Array, Array, Array]]:
        nonlocal boundary_topology
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
        if boundary_topology is None:
            boundary_topology = _build_boundary_topology(triangles)
        else:
            boundary_topology.validate_rigid_pose(triangles)
        return (
            q,
            qd,
            triangles,
            _surface_sample(
                triangles, centre, radius, boundary_topology=boundary_topology
            ),
        )

    def clearance(data: tuple[float, float, int, Array, Array, Array]) -> float:
        return float(np.linalg.norm(centre - data[3])) - radius

    def physical_approach(q: Array, qd: Array, point: Array, normal: Array) -> float:
        if face_point_velocity_world is None:
            return 0.0
        velocity = np.asarray(face_point_velocity_world(q, qd, point), dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("face point velocity must be a finite three-vector")
        return float(velocity @ normal)

    def local_gap_derivative(time_s: float) -> float:
        """Estimate the finite-surface clearance derivative using the real advance."""
        base = max(duration * 1e-6, min(4.0 * tolerance, duration * 1e-4))
        available_before, available_after = time_s, duration - time_s
        if time_s <= 0.0:
            h = min(base, available_after / 2.0)
            gap0 = clearance(sample(time_s)[3])
            gap_half = clearance(sample(time_s + h / 2.0)[3])
            gap_full = clearance(sample(time_s + h)[3])
            half_slope = (gap_half - gap0) / (h / 2.0)
            full_slope = (gap_full - gap0) / h
            return 2.0 * half_slope - full_slope
        if time_s >= duration:
            h = min(base, available_before / 2.0)
            gap0 = clearance(sample(time_s)[3])
            gap_half = clearance(sample(time_s - h / 2.0)[3])
            gap_full = clearance(sample(time_s - h)[3])
            half_slope = (gap0 - gap_half) / (h / 2.0)
            full_slope = (gap0 - gap_full) / h
            return 2.0 * half_slope - full_slope
        if available_before < base:
            h = available_before / 2.0
            gap0 = clearance(sample(time_s)[3])
            gap_half = clearance(sample(time_s - h / 2.0)[3])
            gap_full = clearance(sample(time_s - h)[3])
            half_slope = (gap0 - gap_half) / (h / 2.0)
            full_slope = (gap0 - gap_full) / h
            return 2.0 * half_slope - full_slope
        if available_after < base:
            h = available_after / 2.0
            gap0 = clearance(sample(time_s)[3])
            gap_half = clearance(sample(time_s + h / 2.0)[3])
            gap_full = clearance(sample(time_s + h)[3])
            half_slope = (gap_half - gap0) / (h / 2.0)
            full_slope = (gap_full - gap0) / h
            return 2.0 * half_slope - full_slope
        h = min(base, available_before, available_after)
        gap_before = clearance(sample(time_s - h)[3])
        gap_after = clearance(sample(time_s + h)[3])
        return (gap_after - gap_before) / (2.0 * h)

    grid_times = np.linspace(0.0, duration, CONTACT_ROOT_SUBDIVISIONS + 1)
    grid_states = [sample(float(time_s)) for time_s in grid_times]
    penalty_samples = np.asarray([state[3][1] for state in grid_states], dtype=float)
    event: ContactRoot | None = None
    if clearance(grid_states[0][3]) == 0.0 and local_gap_derivative(0.0) < 0.0:
        q_star, qd_star, triangles, data_star = grid_states[0]
        index, point, bary = data_star[2], data_star[3], data_star[4]
        delta = centre - point
        distance = float(np.linalg.norm(delta))
        approach = (
            physical_approach(q_star, qd_star, point, delta / distance)
            if distance > 0
            else 0.0
        )
        if (
            distance > 0.0
            and approach > MIN_APPROACH_SPEED_MPS
            and float(delta @ data_star[5]) > 0.0
            and not _boundary_edge_contact(
                triangles, index, bary, topology=boundary_topology
            )
        ):
            event = ContactRoot(
                0.0,
                q_star.copy(),
                qd_star.copy(),
                point.copy(),
                delta / distance,
                approach,
                int(index),
            )

    brackets = []
    if event is None:
        for bracket_index in range(CONTACT_ROOT_SUBDIVISIONS):
            lo, hi = map(float, grid_times[bracket_index : bracket_index + 2])
            data_lo, data_hi = (
                grid_states[i][3] for i in (bracket_index, bracket_index + 1)
            )
            if clearance(data_lo) > 0.0 and clearance(data_hi) <= 0.0:
                brackets.append((lo, hi, data_lo, data_hi))
            elif data_lo[0] > 0.0 and data_hi[0] <= 0.0:
                times = np.linspace(lo, hi, SIDE_SWITCH_REFINEMENT + 1)
                previous_time, previous = lo, data_lo
                for current_time in times[1:]:
                    current = (
                        data_hi
                        if current_time == hi
                        else sample(float(current_time))[3]
                    )
                    if clearance(previous) > 0.0 and clearance(current) <= 0.0:
                        brackets.append(
                            (previous_time, float(current_time), previous, current)
                        )
                    previous_time, previous = float(current_time), current
        for lo, hi, data_lo, data_hi in brackets:
            exact_right_root = clearance(data_hi) == 0.0
            if exact_right_root:
                time_s = hi
            else:
                for _ in range(max_iterations):
                    mid = 0.5 * (lo + hi)
                    _, _, _, data_mid = sample(mid)
                    gap_mid = clearance(data_mid)
                    if hi - lo <= tolerance and abs(gap_mid) <= distance_tolerance:
                        time_s = mid
                        break
                    if gap_mid > 0.0:
                        lo, data_lo = mid, data_mid
                    else:
                        hi, data_hi = mid, data_mid
                else:
                    raise ValueError(
                        "contact root did not converge within max_iterations"
                    )
            q_star, qd_star, triangles, data_star = sample(time_s)
            index, point, bary = data_star[2], data_star[3], data_star[4]
            delta = centre - point
            distance = float(np.linalg.norm(delta))
            if (
                distance == 0.0
                or float(delta @ data_star[5]) <= 0.0
                or _boundary_edge_contact(
                    triangles, index, bary, topology=boundary_topology
                )
            ):
                continue
            gap_derivative = local_gap_derivative(time_s)
            if not math.isfinite(gap_derivative) or gap_derivative >= 0.0:
                continue
            approach = physical_approach(q_star, qd_star, point, delta / distance)
            if not math.isfinite(approach) or approach <= MIN_APPROACH_SPEED_MPS:
                continue
            event = ContactRoot(
                time_s,
                q_star.copy(),
                qd_star.copy(),
                point.copy(),
                delta / distance,
                approach,
                int(index),
            )
            break

    penalty = -radius * float(
        logsumexp(-penalty_samples / radius) - math.log(len(penalty_samples))
    )
    # A bracket proves only that the sampled trajectory contains a candidate
    # bracket. No bracket does not prove absence between the fixed samples.
    status = (
        "sampled_bracket_candidate"
        if event is not None
        else "unresolved_no_accepted_candidate"
    )
    return ContactSearchResult(event, penalty, status)
