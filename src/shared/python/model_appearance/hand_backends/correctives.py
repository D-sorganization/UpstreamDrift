"""Corrective shapes applied on top of linear blend skinning (epic #15150).

Joint rotation with linear blend skinning cannot do two things a real hand
does: keep the skin of a web from folding through itself, and let the soft
pads of the palm settle onto what the hand holds. The correctives here move
vertices *after* skinning to supply that, and lift skin that ends a hair
inside the grip back to its surface. All are owner-approved (2026-11-15),
visual only, and switchable: zero iterations, zero reach or zero depth
leaves the skinned mesh untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
FoldedFaces = Callable[[Array], IntArray]
Distance = Callable[[Array], Array]
_GRADIENT_STEP_M = 1e-5


@dataclass(frozen=True)
class FoldRelaxSettings:
    """Laplacian relaxation of the skin around folded triangles.

    ``iterations`` is the most passes to run (``0`` switches it off),
    ``rings`` how many edge rings around a folded triangle relax with it and
    ``step`` the share of the way each vertex moves to its neighbours' mean.
    """

    iterations: int
    rings: int
    step: float

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.rings < 0:
            raise ValueError("fold relax iterations and rings must not be negative")
        if not 0.0 < self.step <= 1.0:
            raise ValueError("fold relax step must be in (0, 1]")


@dataclass(frozen=True)
class PadConformSettings:
    """Soft pads settling onto the obstacle the hand holds.

    Pad skin that faces the obstacle and lies within ``reach_m`` of it moves
    toward it, by most where it is already close and by nothing at the edge
    of the reach (``reach_m == 0`` switches it off). ``clearance_m`` is the
    gap left at full contact, ``facing_full`` the cosine between skin normal
    and approach direction at which a vertex moves in full, and
    ``smooth_iterations`` how often the displacement field is averaged over
    neighbours so the pad stays smooth, and ``passes`` how often the whole
    step repeats (smoothing spreads each pass, so one pass stops short).
    """

    bones: tuple[str, ...]
    reach_m: float
    clearance_m: float
    facing_full: float
    smooth_iterations: int
    passes: int

    def __post_init__(self) -> None:
        if self.reach_m < 0.0 or self.clearance_m < 0.0:
            raise ValueError("pad reach and clearance must not be negative")
        if not 0.0 < self.facing_full <= 1.0:
            raise ValueError("pad facing_full must be in (0, 1]")
        if self.smooth_iterations < 0 or self.passes < 0:
            raise ValueError("pad smooth_iterations and passes must not be negative")


@dataclass(frozen=True)
class PressSettings:
    """Soft skin pressed against the obstacle the hand squeezes.

    Skin up to ``max_depth_m`` inside the obstacle is laid on its surface
    (``0`` switches the corrective off). The flesh it displaces bulges beside
    the contact patch: skin within ``bulge_rings`` edges of pressed skin
    moves sideways by ``bulge_gain`` times the press depth spread to it
    (``bulge_gain == 0`` leaves that skin alone).
    """

    max_depth_m: float
    bulge_gain: float
    bulge_rings: int

    def __post_init__(self) -> None:
        if self.max_depth_m < 0.0 or self.bulge_gain < 0.0 or self.bulge_rings < 0:
            raise ValueError("press depth, bulge gain and rings must not be negative")


def vertex_neighbours(faces: IntArray, vertex_count: int) -> list[IntArray]:
    """Edge-connected neighbours of every vertex."""
    pairs = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    ).astype(np.int64)
    both = np.unique(np.concatenate([pairs, pairs[:, ::-1]]), axis=0)
    starts = np.searchsorted(both[:, 0], np.arange(vertex_count + 1))
    return [both[starts[i] : starts[i + 1], 1] for i in range(vertex_count)]


def _neighbour_mean(values: Array, neighbours: list[IntArray], ids: IntArray) -> Array:
    return np.array([values[neighbours[i]].mean(axis=0) for i in ids])


def _grown(seed: IntArray, neighbours: list[IntArray], rings: int) -> IntArray:
    region = {int(i) for i in seed}
    for _ in range(rings):
        region |= {int(j) for i in region for j in neighbours[i]}
    return np.array(sorted(region), dtype=np.int64)


def relax_folds(
    vertices: Array, faces: IntArray, folded: FoldedFaces, settings: FoldRelaxSettings
) -> Array:
    """Relax the skin around folded triangles until none is left.

    ``folded(vertices)`` returns the ids of the triangles that are inside
    out. Each pass moves the vertices within ``settings.rings`` of such a
    triangle part of the way to the mean of their neighbours. Postcondition:
    vertices outside every relaxed region are returned unchanged.
    """
    out = np.array(vertices, dtype=np.float64)
    if settings.iterations == 0:
        return out
    neighbours = vertex_neighbours(faces, len(out))
    for _ in range(settings.iterations):
        bad = folded(out)
        if len(bad) == 0:
            break
        region = _grown(np.unique(faces[bad]), neighbours, settings.rings)
        mean = _neighbour_mean(out, neighbours, region)
        out[region] += settings.step * (mean - out[region])
    return out


def distance_gradient(distance: Distance, points: Array, step_m: float) -> Array:
    """Unit direction of steepest increase of ``distance`` (central difference)."""
    grad = np.empty_like(points)
    for k in range(3):
        offset = np.zeros(3)
        offset[k] = step_m
        grad[:, k] = distance(points + offset) - distance(points - offset)
    norm = np.linalg.norm(grad, axis=1, keepdims=True)
    return np.divide(grad, norm, out=np.zeros_like(grad), where=norm > 0.0)


def _smoothstep(x: Array) -> Array:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _conform_once(
    out: Array,
    normals: Array,
    neighbours: list[IntArray],
    pad_ids: IntArray,
    distance: Distance,
    settings: PadConformSettings,
) -> None:
    pads = out[pad_ids]
    gap = distance(pads)
    away = distance_gradient(distance, pads, _GRADIENT_STEP_M)
    facing = _smoothstep(
        -np.einsum("ij,ij->i", normals[pad_ids], away) / settings.facing_full
    )
    near = 1.0 - np.clip(gap / settings.reach_m, 0.0, 1.0) ** 2
    travel = np.maximum(gap - settings.clearance_m, 0.0) * near * facing
    shift = np.zeros_like(out)
    shift[pad_ids] = -travel[:, None] * away
    for _ in range(settings.smooth_iterations):
        shift[pad_ids] = 0.5 * (
            shift[pad_ids] + _neighbour_mean(shift, neighbours, pad_ids)
        )
    moved = pads + shift[pad_ids]
    floor = np.minimum(gap, settings.clearance_m)
    short = np.maximum(floor - distance(moved), 0.0)
    moved += short[:, None] * distance_gradient(distance, moved, _GRADIENT_STEP_M)
    out[pad_ids] = moved


def conform_pads(
    vertices: Array,
    faces: IntArray,
    pad_ids: IntArray,
    distance: Distance,
    settings: PadConformSettings,
    vertex_normals: Callable[[Array], Array],
) -> Array:
    """Settle pad skin onto the obstacle (see :class:`PadConformSettings`).

    ``vertex_normals(vertices)`` returns outward unit normals. Postconditions:
    only ``pad_ids`` move; no moved vertex ends closer to the obstacle than
    ``clearance_m`` unless it already was.
    """
    out = np.array(vertices, dtype=np.float64)
    if settings.reach_m == 0.0 or settings.passes == 0 or len(pad_ids) == 0:
        return out
    neighbours = vertex_neighbours(faces, len(out))
    for _ in range(settings.passes):
        _conform_once(out, vertex_normals(out), neighbours, pad_ids, distance, settings)
    return out


def lift_out(vertices: Array, distance: Distance, *, max_depth_m: float) -> Array:
    """Lift skin that sits just inside the obstacle back to its surface.

    Joints are solved until skin vertices touch, so skin between two touching
    vertices can dip a fraction of a millimetre inside a round grip; flesh
    would compress instead. Vertices up to ``max_depth_m`` inside move out
    along the distance gradient (``0`` switches it off). Deeper ones are a
    posing fault and are left for the quality checks to report.
    """
    if max_depth_m < 0.0:
        raise ValueError("lift-out max_depth_m must not be negative")
    out = np.array(vertices, dtype=np.float64)
    if max_depth_m == 0.0:
        return out
    gap = distance(out)
    ids = np.flatnonzero((gap < 0.0) & (gap >= -max_depth_m))
    if len(ids):
        away = distance_gradient(distance, out[ids], _GRADIENT_STEP_M)
        out[ids] -= gap[ids, None] * away
    return out


def yield_to_points(
    vertices: Array,
    normals_of: Callable[[Array], Array],
    points: Array,
    *,
    max_depth_m: float,
    passes: int,
) -> Array:
    """Dent skin that points of another body poke through.

    The obstacle correctives clear this skin's *vertices*. Where its
    triangles are large, the other body can still stand through the skin
    between them. Each of ``points`` up to ``max_depth_m`` under the skin
    pushes the three skin vertices nearest it inward along their normals by
    its depth, as flesh would give way; zero depth or passes switches it off.
    """
    if max_depth_m < 0.0 or passes < 0:
        raise ValueError("yield max_depth_m and passes must not be negative")
    out = np.array(vertices, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if max_depth_m == 0.0 or len(points) == 0:
        return out
    from scipy.spatial import cKDTree

    for _ in range(passes):
        normals = normals_of(out)
        tree = cKDTree(out)
        reach, nearest = tree.query(points, distance_upper_bound=4.0 * max_depth_m)
        near = np.flatnonzero(np.isfinite(reach))
        first = nearest[near]
        depth = -np.einsum("ij,ij->i", points[near] - out[first], normals[first])
        under = (depth > 0.0) & (depth <= max_depth_m)
        if not under.any():
            break
        ids = tree.query(points[near[under]], k=3)[1]
        dent = np.zeros(len(out))
        np.maximum.at(dent, ids.ravel(), np.repeat(depth[under], 3))
        out -= dent[:, None] * normals
    return out


def press_flat(
    vertices: Array,
    faces: IntArray,
    distance: Distance,
    settings: PressSettings,
    vertex_normals: Callable[[Array], Array],
) -> Array:
    """Lay pressed skin on the obstacle and bulge the skin beside it.

    See :class:`PressSettings`. The bulge moves a vertex along its outward
    normal with the part that points into the obstacle removed, so it spreads
    sideways. Postconditions: skin that was up to ``max_depth_m`` inside ends
    on the surface; no bulged vertex ends inside; skin farther than
    ``bulge_rings`` edges from pressed skin is returned unchanged.
    """
    out = lift_out(vertices, distance, max_depth_m=settings.max_depth_m)
    depth = np.linalg.norm(out - vertices, axis=1)
    if settings.bulge_gain == 0.0 or settings.bulge_rings == 0 or not depth.any():
        return out
    neighbours = vertex_neighbours(faces, len(out))
    pressed = np.flatnonzero(depth > 0.0)
    spread, region = depth.copy(), pressed
    for _ in range(settings.bulge_rings):
        region = _grown(region, neighbours, 1)
        spread[region] = np.maximum(
            spread[region], 0.5 * _neighbour_mean(spread, neighbours, region)
        )
    beside = np.setdiff1d(region, pressed)
    if len(beside) == 0:
        return out
    normal = vertex_normals(out)[beside]
    away = distance_gradient(distance, out[beside], _GRADIENT_STEP_M)
    into = np.minimum(np.einsum("ij,ij->i", normal, away), 0.0)
    sideways = normal - into[:, None] * away
    moved = out[beside] + settings.bulge_gain * spread[beside, None] * sideways
    short = np.maximum(np.minimum(distance(out[beside]), 0.0) - distance(moved), 0.0)
    moved += short[:, None] * distance_gradient(distance, moved, _GRADIENT_STEP_M)
    out[beside] = moved
    return out


def edge_folds(vertices: Array, faces: IntArray) -> tuple[Array, IntArray]:
    """Fold angle (degrees between the two face normals) of every interior
    edge, and the two faces that meet there."""
    sides = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    order = np.lexsort((sides[:, 1], sides[:, 0]))
    sides, face = sides[order], np.tile(np.arange(len(faces)), 3)[order]
    paired = np.flatnonzero((sides[:-1] == sides[1:]).all(axis=1))
    pair = np.column_stack([face[paired], face[paired + 1]])
    normal = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    cosine = np.einsum("ij,ij->i", normal[pair[:, 0]], normal[pair[:, 1]])
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), pair


def sharpened_faces(
    posed: Array, rest: Array, faces: IntArray, max_added_fold_deg: float
) -> IntArray:
    """Faces at an edge that folds more than ``max_added_fold_deg`` beyond
    its fold at rest.

    Skinning pinches the skin between two knuckles into a crease that is not
    inside out, so fold relaxation leaves it, but it catches the light. An
    edge that is sharp at rest (a nail) is not reported. Use it as the
    ``folded`` argument of :func:`relax_folds`.
    """
    if max_added_fold_deg <= 0.0:
        raise ValueError("max_added_fold_deg must be positive")
    now, pair = edge_folds(posed, faces)
    before, _ = edge_folds(rest, faces)
    return np.unique(pair[now - before > max_added_fold_deg]).astype(np.int64)
