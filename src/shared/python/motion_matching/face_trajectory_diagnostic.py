"""Bounded sampled trajectory diagnostic; never certifies the first contact."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from src.shared.python.motion_matching.face_clearance import (
    FaceClearance,
    face_clearance,
    require_initial_separation,
)

Array = NDArray[np.float64]
Advance = Callable[[float], tuple[Array, Array]]
Faces = Callable[[Array], Array]
Velocity = Callable[[Array, Array, Array], Array]


@dataclass(frozen=True)
class Candidate:
    time_s: float
    q: Array
    qd: Array
    gap_m: float
    approach_mps: float


@dataclass(frozen=True)
class Diagnostic:
    status: Literal["unqualified"]
    coverage: Literal["uncertified"]
    sampled_min_gap_m: float
    candidate: Candidate | None


def diagnose(
    advance: Advance,
    faces: Faces,
    face_velocity: Velocity,
    centre_m: Array,
    radius_m: float,
    duration_s: float,
    *,
    samples: int = 9,
    time_tolerance_s: float = 1e-9,
    distance_tolerance_m: float = 1e-8,
) -> Diagnostic:
    """Inspect one sampled interval, returning only an unqualified candidate.

    Preconditions: positive initial gap, finite advancing states and a face
    material-velocity callback. The sampled bracket is not a coverage proof;
    hidden crossings, mesh seam/edge hits and outside-solid placement remain
    unresolved even if a candidate is returned.
    """
    if (
        not math.isfinite(duration_s)
        or duration_s <= 0
        or isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples < 2
        or not math.isfinite(time_tolerance_s)
        or not 0 < time_tolerance_s < duration_s
        or not math.isfinite(distance_tolerance_m)
        or distance_tolerance_m <= 0
    ):
        raise ValueError("invalid interval, samples or tolerances")
    centre = np.asarray(centre_m, dtype=float)
    shapes: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    def evaluate(t: float) -> tuple[Array, Array, Array, FaceClearance]:
        nonlocal shapes
        q, qd = (np.asarray(value, dtype=float) for value in advance(t))
        if q.ndim != 1 or qd.ndim != 1 or not q.size or not qd.size:
            raise ValueError("advance must return nonempty vector states")
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("advance states must be finite")
        current = (q.shape, qd.shape)
        if shapes is not None and current != shapes:
            raise ValueError("advance state dimensions changed")
        shapes = current
        mesh = np.asarray(faces(q), dtype=float)
        sample = face_clearance(mesh, centre, radius_m)
        return q.copy(), qd.copy(), mesh, sample

    q0, _, mesh0, _ = evaluate(0.0)
    del q0
    require_initial_separation(mesh0, centre, radius_m)
    previous_t = 0.0
    previous_gap = face_clearance(mesh0, centre, radius_m).gap_m
    min_gap = previous_gap
    bracket: tuple[float, float] | None = None
    for t in np.linspace(0.0, duration_s, samples)[1:]:
        gap = evaluate(float(t))[3].gap_m
        min_gap = min(min_gap, gap)
        if previous_gap > 0 and gap <= 0:
            bracket = (previous_t, float(t))
            break
        previous_t, previous_gap = float(t), gap
    if bracket is None:
        return Diagnostic("unqualified", "uncertified", min_gap, None)

    root = brentq(
        lambda t: evaluate(float(t))[3].gap_m, *bracket, xtol=time_tolerance_s
    )
    q, qd, mesh, sample = evaluate(float(root))
    point = np.asarray(sample.closest_point_m)
    tri = mesh[sample.triangle_index]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    normal /= np.linalg.norm(normal)
    delta = centre - point
    distance = float(np.linalg.norm(delta))
    v = np.asarray(face_velocity(q, qd, point), dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("face velocity must be a finite three-vector")
    approach = float(v @ delta / distance) if distance > 0 else 0.0
    # Triangle boundary/seam ambiguity is intentionally not admitted here.
    a, b = tri[1] - tri[0], tri[2] - tri[0]
    offset = point - tri[0]
    aa, ab, bb = float(a @ a), float(a @ b), float(b @ b)
    denom = aa * bb - ab * ab
    if not math.isfinite(denom) or denom <= 0:
        raise ValueError("face triangle has unresolved barycentric coordinates")
    u = (bb * float(offset @ a) - ab * float(offset @ b)) / denom
    vcoord = (aa * float(offset @ b) - ab * float(offset @ a)) / denom
    interior = min(u, vcoord, 1 - u - vcoord) > 0
    candidate = None
    if (
        abs(sample.gap_m) <= distance_tolerance_m
        and float(normal @ delta) > 0
        and approach > 0
        and interior
    ):
        candidate = Candidate(float(root), q, qd, sample.gap_m, approach)
    return Diagnostic("unqualified", "uncertified", min_gap, candidate)
