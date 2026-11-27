"""Baseball cap for the CC0 mesh head (visual only, off by default, #15276).

The crown is a thin closed shell that follows the scalp above a cap line (low at
the back, high at the front) and stands off the skin by ``offset``; the hair
under it is cut flatter than that (``CapCrop.hair_under``), so nothing pokes
through. Hair shows below the cap line at the back and sides. The brim is a
closed, slightly drooping plate grown forward from the front of the cap line.
All lengths are head-length units; ``brim_length`` is the user setting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import Array, Mesh, ellipsoid_mesh
from src.shared.python.model_appearance.schema import CAP_BRIM_RANGE
from src.shared.python.model_appearance.head_hair import (
    HairCrop,
    _clip_above,
    _vertex_normals,
    closed_shell,
    open_rim,
)


@dataclass(frozen=True)
class CapCrop:
    """Cap geometry in head lengths (azimuth in radians from the nose)."""

    #: ``(azimuth, cap line height)`` control points, nose (0) to nape (pi).
    line: tuple[tuple[float, float], ...] = (
        (0.00, 0.76),
        (0.90, 0.72),
        (1.80, 0.64),
        (2.60, 0.58),
        (math.pi, 0.57),
    )
    centre_x: float = 0.10  # azimuth origin on the head's x axis
    offset: float = 0.046  # crown outer surface above the skin
    wall: float = 0.006  # crown wall thickness
    brim_half_angle: float = 0.80  # the brim spans +-this azimuth
    brim_forward: float = 0.85  # 1 = straight ahead, 0 = radial from the head
    brim_end_taper: float = 0.25  # relative length at the brim's ends
    brim_rows: int = 4
    brim_root_inset: float = 0.030  # brim starts this far inside the cap line
    brim_thickness: float = 0.010
    brim_droop: float = 0.12  # tip drop per unit of brim length
    front_overhang: float = 0.0  # smooth forward crown projection, head lengths
    hair_under: HairCrop = field(
        default_factory=lambda: HairCrop(
            top_thickness=0.028,
            fringe_thickness=0.0,
            comb_amplitude=0.0,
            heavy_side_gain=0.0,
        )
    )


def flat_cap_crop() -> CapCrop:
    """Return a close fitting, low crown with a short, broad flat-cap peak.

    The reduced crown stand-off and the matching short-hair crop keep hair
    under the shell. The higher cap line lowers the crown edge, while the peak
    has less forward rake than the baseball-cap default.
    """
    return CapCrop(
        line=(
            (0.0, 0.80),
            (0.90, 0.76),
            (1.80, 0.69),
            (2.60, 0.65),
            (math.pi, 0.64),
        ),
        offset=0.034,
        brim_half_angle=0.93,
        brim_forward=0.68,
        brim_end_taper=0.34,
        front_overhang=0.12,
        hair_under=HairCrop(
            top_thickness=0.020,
            fringe_thickness=0.0,
            comb_amplitude=0.0,
            heavy_side_gain=0.0,
        ),
    )


def newsboy_cap_crop() -> CapCrop:
    """Return a fuller period crown over the same closely cropped hair."""
    return replace(flat_cap_crop(), offset=0.085, front_overhang=0.16)


def build_period_cap(
    skin: Mesh, length: float, crop: CapCrop
) -> list[tuple[str, Mesh]]:
    """Add a small crown-to-peak snap to the closed flat-cap assembly."""
    parts = build_cap(skin, length, crop)
    crown, brim = (part.vertices for _, part in parts)
    crown = crown[(np.abs(crown[:, 1]) < 0.035) & (crown[:, 0] > 0.3)]
    brim = brim[(np.abs(brim[:, 1]) < 0.035) & (brim[:, 0] > 0.3)]
    distances = np.linalg.norm(crown[:, None] - brim[None, :], axis=2)
    i, j = np.unravel_index(distances.argmin(), distances.shape)
    centre = (crown[i] + brim[j]) / 2
    axis = crown[i] - brim[j]
    length = np.linalg.norm(axis)
    if length < 1e-9:
        axis = np.array([0.0, 0.0, 1.0])
    snap = ellipsoid_mesh(
        centre,
        np.array([max(length / 2, 0.002), 0.012, 0.006]),
        axis,
        width_hint=np.array([0.0, 1.0, 0.0]),
    )
    return [*parts, ("cap_snap", snap)]


def cap_line_height(crop: CapCrop, azimuth: Array) -> Array:
    """Cap line height at azimuths (radians); mirrors about the midline."""
    pts = np.asarray(crop.line, dtype=float)
    return np.interp(np.abs(azimuth), pts[:, 0], pts[:, 1])


def _brim(rim: Array, crop: CapCrop, length: float) -> Mesh:
    """Closed drooping plate grown forward from the ordered front rim points."""
    ni, nj = len(rim), crop.brim_rows
    az = np.arctan2(rim[:, 1], rim[:, 0] - crop.centre_x)
    radial = np.stack([np.cos(az), np.sin(az), np.zeros(ni)], axis=1)
    forward = np.array([1.0, 0.0, 0.0])
    direction = crop.brim_forward * forward + (1.0 - crop.brim_forward) * radial
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    # Oval plan: the brim is longest on the midline and tapers toward its ends.
    taper = np.sqrt(
        np.clip(1.0 - (az / crop.brim_half_angle) ** 2, crop.brim_end_taper**2, 1.0)
    )
    ahead = np.linspace(0.0, 1.0, nj + 1)
    reach = (
        -crop.brim_root_inset * (ahead[None, :] == 0.0)
        + length * taper[:, None] * ahead[None, :]
    )
    drop = -crop.brim_droop * length * (ahead[None, :] * taper[:, None]) ** 2
    mid = rim[:, None, :] + direction[:, None, :] * reach[:, :, None]
    mid[:, :, 2] += drop
    half = np.array([0.0, 0.0, 0.5 * crop.brim_thickness])
    top, bottom = mid + half, mid - half
    idx = np.arange(ni * (nj + 1)).reshape(ni, nj + 1)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]

    def quads(
        p: NDArray[np.int64],
        q: NDArray[np.int64],
        r: NDArray[np.int64],
        s: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Triangulate the ordered row-to-row plate cells."""
        return np.concatenate(
            [
                np.stack([p, q, r], -1).reshape(-1, 3),
                np.stack([p, r, s], -1).reshape(-1, 3),
            ]
        )

    count = ni * (nj + 1)
    up = quads(a, b, c, d)
    edges = np.concatenate([up[:, [0, 1]], up[:, [1, 2]], up[:, [2, 0]]])
    key = edges[:, 0] * count + edges[:, 1]
    free = edges[~np.isin(key, edges[:, 1] * count + edges[:, 0])]
    u, w = free[:, 0], free[:, 1]  # the walls run against the top boundary
    walls = np.concatenate(
        [np.stack([w, u, u + count], 1), np.stack([w, u + count, w + count], 1)]
    )
    verts = np.vstack([top.reshape(-1, 3), bottom.reshape(-1, 3)])
    faces = np.vstack([up, quads(a, d, c, b) + count, walls]).astype(np.int64)
    tri = verts[faces]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        faces = faces[:, ::-1]
    return Mesh(verts, np.ascontiguousarray(faces))


def build_cap(
    skin: Mesh, brim_length: float, crop: CapCrop | None = None
) -> list[tuple[str, Mesh]]:
    """``[(name, mesh)]`` for the crown and the brim, unit head-length space.

    Preconditions: ``skin`` is the closed outward-wound head mesh; the brim
    length lies in :data:`CAP_BRIM_RANGE`. Postconditions: both meshes are
    closed; the crown stands ``offset`` above the skin, so hair thinner than
    ``offset - wall`` cannot poke through it.
    """
    lo, hi = CAP_BRIM_RANGE
    if not math.isfinite(brim_length) or not lo <= brim_length <= hi:
        raise ValueError(
            f"cap brim length must be within {CAP_BRIM_RANGE}, got {brim_length!r}"
        )
    cfg = crop or CapCrop()
    v0, n0 = skin.vertices, _vertex_normals(skin)
    az0 = np.arctan2(v0[:, 1], v0[:, 0] - cfg.centre_x)
    s0 = v0[:, 2] - cap_line_height(cfg, az0)
    v, n, faces = _clip_above(v0, n0, skin.faces, s0)
    if len(faces) == 0:
        raise ValueError("cap covers no scalp faces; check the cap line")
    crown = closed_shell(
        v, n, faces, np.full(len(v), cfg.wall), cfg.offset - cfg.wall, cfg.wall
    )
    if cfg.front_overhang:
        # Move both shell surfaces together toward the forehead, preserving
        # their stand-off. Smooth gates leave the cap line and crown apex fixed.
        crown_vertices = crown.vertices.copy()
        z = crown_vertices[:, 2]
        front = np.clip((crown_vertices[:, 0] - cfg.centre_x) / 0.24, 0.0, 1.0)
        rise = np.clip((z - 0.79) / 0.09, 0.0, 1.0)
        apex = np.clip((0.99 - z) / 0.07, 0.0, 1.0)
        front, rise, apex = (
            gate * gate * (3.0 - 2.0 * gate) for gate in (front, rise, apex)
        )
        crown_vertices[:, 0] += cfg.front_overhang * front * rise * apex
        crown = Mesh(crown_vertices, crown.faces)
    rim = open_rim(faces)
    pts = v[rim] + (cfg.offset) * n[rim]
    az = np.arctan2(pts[:, 1], pts[:, 0] - cfg.centre_x)
    front_mask = np.abs(az) <= cfg.brim_half_angle
    order = np.argsort(az[front_mask])
    brim_pts = pts[front_mask][order]
    if len(brim_pts) < 3:
        raise ValueError("cap line leaves no front rim for the brim")
    return [("cap_crown", crown), ("cap_brim", _brim(brim_pts, cfg, brim_length))]


def straw_hat_crop() -> CapCrop:
    """Return a rounded straw crown with clearance for short cropped hair."""
    return CapCrop(
        line=((0.0, 0.73), (math.pi, 0.73)),
        offset=0.075,
        hair_under=HairCrop(
            top_thickness=0.020,
            fringe_thickness=0.0,
            comb_amplitude=0.0,
            heavy_side_gain=0.0,
        ),
    )


def _straw_crown_shape(mesh: Mesh, rise: float) -> Mesh:
    """Raise the upper crown or band together into a broad flat top."""
    vertices = mesh.vertices.copy()
    t = np.clip((vertices[:, 2] - 0.82) / rise, 0.0, 1.0)
    high = vertices[:, 2] > 0.82
    vertices[high, 2] = 0.82 + rise * (0.15 * t[high] + 0.85 * (1 - (1 - t[high]) ** 3))
    return Mesh(vertices, mesh.faces)


def build_straw_hat(skin: Mesh) -> list[tuple[str, Mesh]]:
    """Build closed rounded crown, circumferential brim and narrow dark band.

    Dimensions are artistic head-length fractions; all are visual only.
    The brim extends 0.32 radially from the crown and is 0.012 thick.
    """
    cfg = straw_hat_crop()
    rounded = build_cap(skin, 0.3, cfg)[0][1]
    rise = rounded.vertices[:, 2].max() - 0.82
    crown = _straw_crown_shape(rounded, rise)
    v0, n0 = skin.vertices, _vertex_normals(skin)
    v, n, faces = _clip_above(v0, n0, skin.faces, v0[:, 2] - 0.73)
    rim = open_rim(faces)
    pts = v[rim] + cfg.offset * n[rim]
    angle = np.arctan2(pts[:, 1], pts[:, 0] - cfg.centre_x)
    pts, angle = pts[np.argsort(angle)], np.sort(angle)
    radial = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(angle))))
    inner = pts.copy()
    outer = pts + 0.32 * radial
    outer[:, 2] -= 0.025
    upper = np.array([0.0, 0.0, 0.006])
    rings = np.stack((inner + upper, outer + upper, outer - upper, inner - upper))
    count = len(pts)
    faces_list: list[NDArray[np.int64]] = []
    for i in range(4):
        a = np.arange(count) + i * count
        b = np.roll(np.arange(count), -1) + i * count
        c = np.roll(np.arange(count), -1) + ((i + 1) % 4) * count
        d = np.arange(count) + ((i + 1) % 4) * count
        faces_list.extend((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)))
    mesh = Mesh(rings.reshape(-1, 3), np.concatenate(faces_list))
    if mesh.volume() < 0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, ::-1].copy())
    vb, nb, fb = _clip_above(v0, n0, skin.faces, v0[:, 2] - 0.76)
    vb, nb, fb = _clip_above(vb, nb, fb, 0.82 - vb[:, 2])
    band = closed_shell(vb, nb, fb, np.full(len(vb), 0.006), cfg.offset, 0.006)
    band = _straw_crown_shape(band, rise)
    return [("straw_crown", crown), ("straw_brim", mesh), ("straw_band", band)]
