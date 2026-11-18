"""Short-crop haircut grown on the CC0 skin mesh (pure numpy, #15276).

The haircut is a closed shell built from the skin's own scalp faces: an inner
surface half a millimetre off the skin, an outer surface offset along the skin
normal by a thickness field, and the two welded along the hairline where the
thickness is zero. Short back and sides come from a hairline that drops from
the forehead through the temples and over the ear to the nape and from a
thickness that grows from a tapered fade at the sides to volume on top, a
swept fringe at the front and an optional parting. Every number is a field of
:class:`HairCrop`; lengths are head lengths (neck plane to vertex = 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import Array, Mesh

_PART_SIGN = {"left": 1.0, "right": -1.0, "none": 0.0}


@dataclass(frozen=True)
class HairCrop:
    """Haircut parameters in head lengths (azimuth in radians from the nose)."""

    #: ``(azimuth, hairline height)`` control points, nose (0) to nape (pi).
    hairline: tuple[tuple[float, float], ...] = (
        (0.00, 0.80),
        (0.50, 0.79),
        (0.85, 0.70),
        (1.10, 0.56),
        (1.60, 0.62),
        (1.85, 0.50),
        (2.05, 0.40),
        (2.45, 0.35),
        (2.85, 0.32),
        (math.pi, 0.31),
    )
    centre_x: float = 0.10  # azimuth origin on the head's x axis
    off_skin: float = 0.0020  # inner surface clearance from the skin
    side_thickness: float = 0.008  # tapered fade at the sides and back
    top_thickness: float = 0.066  # length left on top
    fade_from_z: float = 0.70  # thickness blends from side to top over this band
    fade_to_z: float = 1.00
    rim_thickness: float = 0.0015  # thin wall at the hairline edge
    edge_band: float = 0.07  # height over which thickness ramps up from the line
    fringe_thickness: float = 0.060  # extra swept fringe at the front hairline
    fringe_x: float = 0.46  # fringe centre along x
    fringe_z: float = 0.85  # fringe centre height
    fringe_width: float = 0.15  # fringe gaussian radius
    comb_amplitude: float = 0.009  # combed ridges running front to back
    comb_wavelength: float = 0.085
    comb_sweep: float = 0.30  # strands lean toward the parted side by this slope
    fringe_band: float = 0.02  # the forelock ramps up faster than the crop
    crown_from_z: float = 0.78  # combing and parting start above this height
    crown_band: float = 0.10
    part_from_x: float = 0.05  # the parting runs from here to the front
    part_front_band: float = 0.15
    part_front_z_band: float = 0.08
    heavy_band: float = 0.05  # width over which the parted side gains volume
    heavy_offset: float = 0.2
    part_y: float = 0.075  # parting offset from the midline
    part_width: float = 0.014
    part_depth: float = 0.35  # fraction of thickness removed in the groove
    heavy_side_gain: float = 0.18  # extra volume swept over the parted side


def _clip_above(
    v: Array, n: Array, faces: NDArray[np.int64], s: Array
) -> tuple[Array, Array, NDArray[np.int64]]:
    """Faces clipped to ``s > 0``, cut exactly where ``s`` crosses zero.

    Returns the vertices and normals extended by the cut points and the new
    triangles (winding preserved). A cut point is shared by the faces on both
    sides of its edge, so the clipped surface stays a manifold with a smooth
    hairline instead of a staircase of whole triangles.
    """
    cuts: dict[tuple[int, int], int] = {}
    new_v: list[Array] = []
    new_n: list[Array] = []

    def cut(inside: int, outside: int) -> int:
        key = (min(inside, outside), max(inside, outside))
        if key not in cuts:
            t = s[inside] / (s[inside] - s[outside])
            pos = v[inside] + t * (v[outside] - v[inside])
            nrm = n[inside] + t * (n[outside] - n[inside])
            cuts[key] = len(v) + len(new_v)
            new_v.append(pos)
            new_n.append(nrm / (np.linalg.norm(nrm) + 1e-12))
        return cuts[key]

    out: list[tuple[int, int, int]] = []
    for face in faces:
        inside = [int(k) for k in range(3) if s[face[k]] > 0.0]
        if len(inside) == 3:
            out.append((face[0], face[1], face[2]))
        elif len(inside) == 1:
            a, b, c = np.roll(face, -inside[0])
            out.append((a, cut(a, b), cut(a, c)))
        elif len(inside) == 2:
            lone = next(k for k in range(3) if k not in inside)
            a, b, c = np.roll(face, -(lone + 1))  # a, b inside; c outside
            pb, pa = cut(b, c), cut(a, c)
            out += [(a, b, pb), (a, pb, pa)]
    verts = np.vstack([v, new_v]) if new_v else v
    norms = np.vstack([n, new_n]) if new_n else n
    return verts, norms, np.asarray(out, dtype=np.int64)


def _vertex_normals(mesh: Mesh) -> Array:
    tri = mesh.vertices[mesh.faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(mesh.vertices)
    for k in range(3):
        np.add.at(out, mesh.faces[:, k], face_n)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def _smooth(x: NDArray[np.float64]) -> NDArray[np.float64]:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hairline_height(crop: HairCrop, azimuth: NDArray[np.float64]) -> Array:
    """Hairline height at azimuths (radians); mirrors about the midline."""
    pts = np.asarray(crop.hairline, dtype=float)
    return np.interp(np.abs(azimuth), pts[:, 0], pts[:, 1])


def _thickness(crop: HairCrop, v: Array, part: str) -> tuple[Array, Array]:
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    line = hairline_height(crop, np.arctan2(y, x - crop.centre_x))
    top = _smooth((z - crop.fade_from_z) / (crop.fade_to_z - crop.fade_from_z))
    t = crop.side_thickness + (crop.top_thickness - crop.side_thickness) * top
    sign = _PART_SIGN[part]
    top_mask = _smooth((z - crop.crown_from_z) / crop.crown_band)
    phase = (y - sign * crop.comb_sweep * (x - crop.centre_x)) / crop.comb_wavelength
    t = t + crop.comb_amplitude * top_mask * np.cos(2 * np.pi * phase)
    r2 = ((x - crop.fringe_x) ** 2 + (z - crop.fringe_z) ** 2) / crop.fringe_width**2
    forelock = (
        crop.fringe_thickness * np.exp(-r2) * _smooth((z - line) / crop.fringe_band)
    )
    if sign:
        groove = np.exp(-(((y - sign * crop.part_y) / crop.part_width) ** 2))
        front = _smooth((x - crop.part_from_x) / crop.part_front_band) * _smooth(
            (z - crop.crown_from_z) / crop.part_front_z_band
        )
        heavy = _smooth(
            sign * (y - sign * crop.part_y) / crop.heavy_band - crop.heavy_offset
        )
        t = t * (1.0 - crop.part_depth * groove * front)
        t = t * (1.0 + crop.heavy_side_gain * heavy * front)
    return t * _smooth((z - line) / crop.edge_band) + forelock, line


def build_short_crop(
    skin: Mesh, part: str = "left", crop: HairCrop | None = None
) -> Mesh:
    """Closed haircut shell over ``skin`` (unit head-length coordinates).

    Preconditions: ``skin`` is a closed outward-wound head mesh in the
    canonical frame (x forward, z up, vertex at 1); ``part`` is ``left``,
    ``right`` or ``none``. Postconditions: the result is a closed manifold of
    positive volume whose every vertex lies on or outside the skin, with zero
    thickness along the hairline; ``part='none'`` mirrors left/right.
    """
    if part not in _PART_SIGN:
        raise ValueError(f"hair part must be one of {tuple(_PART_SIGN)}, got {part!r}")
    cfg = crop or HairCrop()
    v0, n0 = skin.vertices, _vertex_normals(skin)
    _, line0 = _thickness(cfg, v0, part)
    v, n, faces = _clip_above(v0, n0, skin.faces, v0[:, 2] - line0)
    if len(faces) == 0:
        raise ValueError("haircut covers no scalp faces; check the hairline")
    thick, _ = _thickness(cfg, v, part)
    return closed_shell(v, n, faces, thick, cfg.off_skin, cfg.rim_thickness)


def open_rim(faces: NDArray[np.int64]) -> NDArray[np.int64]:
    """Sorted vertex indices on the open boundary of a triangle surface."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    size = int(edges.max()) + 1
    key = edges[:, 0] * size + edges[:, 1]
    twin = edges[:, 1] * size + edges[:, 0]
    return np.unique(edges[~np.isin(key, twin)])


def closed_shell(
    v: Array,
    n: Array,
    faces: NDArray[np.int64],
    thick: Array,
    off_skin: float,
    rim_thickness: float,
) -> Mesh:
    """Closed shell over the open surface ``faces``: outer, inner and rim walls.

    The inner surface is ``off_skin`` along the normals, the outer one
    ``thick`` further; along the open boundary the thickness is
    ``rim_thickness``. Postcondition: closed manifold, outward winding.
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    size = len(v)
    key = edges[:, 0] * size + edges[:, 1]
    twin = edges[:, 1] * size + edges[:, 0]
    open_edge = ~np.isin(key, twin)
    rim = np.unique(edges[open_edge])
    thick = thick.copy()
    thick[rim] = rim_thickness
    inner = v + off_skin * n
    outer = inner + thick[:, None] * n
    used = np.unique(faces)
    remap = np.full(size, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    count = len(used)
    # Outer skin, inner skin (reversed) and a thin wall along every open edge
    # a -> b: it closes the shell even where the rim pinches or has chords.
    a, b = remap[edges[open_edge, 0]], remap[edges[open_edge, 1]]
    wall = np.concatenate(
        [np.stack([b, a, a + count], 1), np.stack([b, a + count, b + count], 1)]
    )
    shell = np.vstack([remap[faces], remap[faces][:, ::-1] + count, wall])
    return Mesh(np.vstack([outer[used], inner[used]]), shell.astype(np.int64))
