"""SDF smooth-union meshing for procedural skin (scikit-image marching cubes)."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from src.shared.python.model_appearance.geometry import Mesh


def _smooth_min_iq(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def sdf_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-18:
        return np.linalg.norm(p - a, axis=1)
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    closest = a + np.outer(t, ab)
    return np.linalg.norm(p - closest, axis=1)


def sdf_tapered_polyline(
    p: np.ndarray,
    points: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Unsigned shell distance (0 on surface, negative inside)."""
    d = np.full(len(p), 1e6, dtype=float)
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        ab = b - a
        denom = max(float(np.dot(ab, ab)), 1e-18)
        t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
        closest = a + np.outer(t, ab)
        seg_dist = np.linalg.norm(p - closest, axis=1)
        r = radii[i] * (1.0 - t) + radii[i + 1] * t
        d = np.minimum(d, seg_dist - r)
    return d


def sdf_rounded_box(
    p: np.ndarray,
    centre: np.ndarray,
    half_extents: np.ndarray,
    corner_radius: float,
) -> np.ndarray:
    q = np.abs(p - centre) - half_extents
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside - corner_radius


def sdf_cupped_palm(
    p: np.ndarray,
    *,
    grip_radius_m: float,
    z_lo: float,
    z_hi: float,
    palm_width_m: float,
    palmar_clearance_m: float,
    cup_depth_m: float,
    transverse_arch_m: float,
    dorsal_lift_m: float,
) -> np.ndarray:
    """Cupped metacarpal slab; negative inside flesh."""
    z_mid = 0.5 * (z_lo + z_hi)
    radial = grip_radius_m + palmar_clearance_m + cup_depth_m * 0.45
    centre = np.array([radial + dorsal_lift_m * 0.35, 0.0, z_mid])
    half = np.array([dorsal_lift_m * 0.65, palm_width_m * 0.5, (z_hi - z_lo) * 0.52])
    d = sdf_rounded_box(p, centre, half, 0.005)
    # Transverse arch: deepen V-centre
    arch = transverse_arch_m * np.exp(
        -((p[:, 2] - z_mid) ** 2) / (2.0 * ((z_hi - z_lo) * 0.35) ** 2)
    )
    d = d - arch * np.exp(-(p[:, 1] ** 2) / (2.0 * (palm_width_m * 0.22) ** 2))
    # Keep palm outside grip bore
    rho = np.hypot(p[:, 0], p[:, 1])
    bore = grip_radius_m - 0.001 - rho
    d = np.maximum(d, bore)
    return d


def sdf_thenar_mount(p: np.ndarray, centre: np.ndarray, radius_m: float) -> np.ndarray:
    return np.linalg.norm(p - centre, axis=1) - radius_m


def combine_sdfs(fields: list[np.ndarray], smooth_k: float) -> np.ndarray:
    out = fields[0]
    for f in fields[1:]:
        out = _smooth_min_iq(out, f, smooth_k)
    return out


def mesh_from_sdf(
    sdf: Callable[[np.ndarray], np.ndarray],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    *,
    resolution: int = 80,
    smooth_k: float = 0.003,
) -> Mesh:
    xs = np.linspace(bounds_min[0], bounds_max[0], resolution)
    ys = np.linspace(bounds_min[1], bounds_max[1], resolution)
    zs = np.linspace(bounds_min[2], bounds_max[2], resolution)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    flat = grid.reshape(-1, 3)
    _ = smooth_k
    field = sdf(flat).reshape(grid.shape[:3])
    spacing = (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise RuntimeError(
            "procedural_skin SDF requires scikit-image (pip install scikit-image)"
        ) from exc
    verts, faces, _, _ = marching_cubes(field, level=0.0, spacing=spacing)
    world = np.column_stack(
        [
            verts[:, 0] + bounds_min[0],
            verts[:, 1] + bounds_min[1],
            verts[:, 2] + bounds_min[2],
        ]
    )
    return Mesh(world.astype(np.float64), faces.astype(np.int64))


def catmull_rom_spline(points: np.ndarray, samples: int) -> np.ndarray:
    if len(points) < 2:
        return points
    if len(points) == 2:
        t_lin = np.linspace(0.0, 1.0, samples)
        return points[0] + np.outer(t_lin, points[1] - points[0])
    pts = [points[0], *points, points[-1]]
    out: list[np.ndarray] = []
    segs = len(points) - 1
    per = max(2, samples // segs)
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for j in range(per):
            t = j / per
            t2, t3 = t * t, t * t * t
            out.append(
                0.5
                * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
            )
    out.append(points[-1])
    return np.asarray(out, dtype=float)


def tube_mesh_from_polyline(
    points: np.ndarray,
    base_radius_m: float,
    tip_radius_m: float,
    *,
    sides: int = 20,
    spline_samples: int = 48,
) -> Mesh:
    from src.shared.python.model_appearance.geometry import _orient

    curve = catmull_rom_spline(points, spline_samples)
    arclen = np.zeros(len(curve))
    for i in range(1, len(curve)):
        arclen[i] = arclen[i - 1] + float(np.linalg.norm(curve[i] - curve[i - 1]))
    total = arclen[-1] if arclen[-1] > 1e-9 else 1.0
    rings: list[np.ndarray] = []
    for i, centre in enumerate(curve):
        frac = arclen[i] / total
        radius = base_radius_m * (1.0 - frac) + tip_radius_m * frac
        if i == 0:
            tangent = curve[1] - curve[0]
        elif i == len(curve) - 1:
            tangent = curve[-1] - curve[-2]
        else:
            tangent = curve[i + 1] - curve[i - 1]
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-12)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(tangent, ref))) > 0.92:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(tangent, ref)
        u /= np.linalg.norm(u)
        v = np.cross(tangent, u)
        phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
        ring = centre + radius * (np.outer(np.cos(phi), u) + np.outer(np.sin(phi), v))
        rings.append(ring)
    verts = np.vstack(rings)
    faces: list[tuple[int, int, int]] = []
    for k in range(len(rings) - 1):
        base = k * sides
        nxt = (k + 1) * sides
        for j in range(sides):
            j2 = (j + 1) % sides
            a, b = base + j, base + j2
            c, d = nxt + j2, nxt + j
            faces += [(a, b, c), (a, c, d)]
    return _orient(verts, np.asarray(faces, dtype=np.int64))


__all__ = [
    "catmull_rom_spline",
    "combine_sdfs",
    "mesh_from_sdf",
    "sdf_cupped_palm",
    "sdf_thenar_mount",
    "sdf_tapered_polyline",
    "tube_mesh_from_polyline",
]
