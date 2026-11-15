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
