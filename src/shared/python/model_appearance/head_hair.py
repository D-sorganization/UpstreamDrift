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
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from src.shared.python.model_appearance.geometry import Array, Mesh
from src.shared.python.model_appearance.head_features import face_settings

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
    cut_steps: int = 0  # bisections placing a cut point on the curved hairline
    refine_splits: int = 0  # rounds of midpoint splits of the scalp under the hair
    refine_edge: float = 0.0  # an edge no longer than this is left whole
    refine_curve: float = 0.75  # share of the normal-curved bulge a midpoint takes


def _turned(faces: NDArray[np.int64], first: NDArray[np.int64]) -> NDArray[np.int64]:
    """Each row rotated so that its column ``first`` leads (winding preserved)."""
    return np.take_along_axis(faces, (first[:, None] + np.arange(3)) % 3, axis=1)


def _split(
    v: Array, n: Array, faces: NDArray[np.int64], cfg: HairCrop
) -> tuple[Array, Array, NDArray[np.int64]]:
    """One round of splits at the midpoints of the edges over ``refine_edge``.

    A midpoint is shared by both faces of its edge, so the surface stays a
    manifold: a face with three long edges becomes four, with two three and
    with one two. The point is lifted along the interpolated normal by
    ``refine_curve`` times its height under the end points' tangent planes,
    outward only, so a convex scalp rounds and nothing sinks into the skin.
    """
    corners = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, index = np.unique(np.sort(corners, axis=1), axis=0, return_inverse=True)
    long = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1) > cfg.refine_edge
    a, b = edges[long, 0], edges[long, 1]
    normal = n[a] + n[b]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    half = 0.5 * (v[b] - v[a])
    sink = np.einsum("ij,ij->i", half, n[a])[:, None] * n[a]
    sink -= np.einsum("ij,ij->i", half, n[b])[:, None] * n[b]
    bulge = np.maximum(-0.5 * np.einsum("ij,ij->i", sink, normal), 0.0)
    mid = 0.5 * (v[a] + v[b]) + cfg.refine_curve * bulge[:, None] * normal
    ident = np.full(len(edges), -1, dtype=np.int64)
    ident[long] = len(v) + np.arange(len(a))
    mids = ident[index].reshape(3, -1).T  # column k: the midpoint of edge k -> k + 1
    marks = (mids >= 0).sum(axis=1)
    one, two, three = marks == 1, marks == 2, marks == 3
    f0, f1, f2 = _turned(faces[one], (mids[one] >= 0).argmax(axis=1)).T
    g0, g1, g2 = _turned(faces[two], ((mids[two] < 0).argmax(axis=1) + 1) % 3).T
    lone = mids[one].max(axis=1)
    gab, gbc, _ = _turned(mids[two], ((mids[two] < 0).argmax(axis=1) + 1) % 3).T
    (h0, h1, h2), (ab, bc, ca) = faces[three].T, mids[three].T
    cut = [(f0, lone, f2), (lone, f1, f2), (g1, gbc, gab), (g0, gab, gbc)]
    cut += [(g0, gbc, g2), (h0, ab, ca), (h1, bc, ab), (h2, ca, bc), (ab, bc, ca)]
    split = np.concatenate([faces[marks == 0]] + [np.stack(c, axis=1) for c in cut])
    return np.vstack([v, mid]), np.vstack([n, normal]), split


def _clip_above(
    v: Array,
    n: Array,
    faces: NDArray[np.int64],
    s: Array,
    exact: tuple[Callable[[Array], float], int] | None = None,
) -> tuple[Array, Array, NDArray[np.int64]]:
    """Faces clipped to ``s > 0``, cut exactly where ``s`` crosses zero.

    Returns the vertices and normals extended by the cut points and the new
    triangles (winding preserved). A cut point is shared by the faces on both
    sides of its edge, so the clipped surface stays a manifold with a smooth
    hairline instead of a staircase of whole triangles. ``s`` is linear along
    an edge only for a straight line; ``exact`` (the level function and a
    number of bisections) puts each cut point on the curved line itself, which
    a 10 mm scalp edge otherwise misses by millimetres (a saw-tooth).
    """
    level, steps = exact or ((lambda _point: 0.0), 0)
    cuts: dict[tuple[int, int], int] = {}
    new_v: list[Array] = []
    new_n: list[Array] = []

    def cut(inside: int, outside: int) -> int:
        key = (min(inside, outside), max(inside, outside))
        if key not in cuts:
            t = s[inside] / (s[inside] - s[outside])
            low, high = 0.0, 1.0
            for _ in range(steps):
                t = 0.5 * (low + high)
                ahead = level(v[inside] + t * (v[outside] - v[inside])) > 0.0
                low, high = (t, high) if ahead else (low, t)
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


def _scalp(
    skin: Mesh, cfg: HairCrop, part: str
) -> tuple[Array, Array, NDArray[np.int64]]:
    """Scalp above the hairline: vertices, normals and faces, cut on the line."""
    if cfg.cut_steps < 0 or cfg.refine_splits < 0:
        raise ValueError("hair cut_steps and refine_splits must not be negative")
    if not 0.0 <= cfg.refine_curve <= 1.0 or not 0.0 <= cfg.refine_edge < 1.0:
        raise ValueError("hair refine_curve and refine_edge must lie in 0..1")
    v, n, faces = skin.vertices, _vertex_normals(skin), skin.faces
    _, line = _thickness(cfg, v, part)
    if cfg.refine_splits:  # only the faces that reach above the line; the face stays
        faces = faces[(v[faces, 2] > line[faces]).any(axis=1)]
        for _ in range(cfg.refine_splits):
            v, n, faces = _split(v, n, faces, cfg)
        _, line = _thickness(cfg, v, part)

    def above(p: Array) -> float:
        azimuth = np.arctan2(p[1], p[0] - cfg.centre_x)
        return float(p[2] - hairline_height(cfg, azimuth))

    exact = (above, cfg.cut_steps) if cfg.cut_steps else None
    v, n, faces = _clip_above(v, n, faces, v[:, 2] - line, exact)
    if len(faces) == 0:
        raise ValueError("haircut covers no scalp faces; check the hairline")
    return v, n, faces


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


@lru_cache(maxsize=8)
def _hairline(points: tuple[tuple[float, float], ...]) -> tuple[Any, Array]:
    pts = np.asarray(points, dtype=float)
    return PchipInterpolator(pts[:, 0], pts[:, 1]), pts[[0, -1], 0]


def load_hair_crop() -> HairCrop:
    """The haircut of ``face_settings.json`` (its ``hair`` section over the defaults)."""
    try:
        return HairCrop(**face_settings("hair"))
    except TypeError as unknown:
        raise ValueError(f"unknown hair setting: {unknown}") from unknown


def hairline_height(crop: HairCrop, azimuth: NDArray[np.float64]) -> Array:
    """Hairline height at azimuths (radians); mirrors about the midline.

    A monotone cubic through the control points: it passes through each one
    without overshoot and has no corner there (#15299).
    """
    line, (first, last) = _hairline(tuple(map(tuple, crop.hairline)))
    return line(np.clip(np.abs(azimuth), first, last))


def hairline_turns_deg(
    skin: Mesh, part: str, step_rad: float, crop: HairCrop | None = None
) -> Array:
    """Turning angles (degrees) along the cut hairline of ``skin``.

    The rim points of the cut are averaged in azimuth bins ``step_rad`` wide
    and the closed line through the bin centres is walked. Preconditions: a
    positive finite step. Postcondition: one angle per occupied bin.
    """
    if not math.isfinite(step_rad) or step_rad <= 0.0:
        raise ValueError("the hairline step must be a positive angle")
    cfg = crop or load_hair_crop()
    v, _, faces = _scalp(skin, cfg, part)
    rim = v[open_rim(faces)]
    bins = np.floor(np.arctan2(rim[:, 1], rim[:, 0] - cfg.centre_x) / step_rad)
    centres = np.array([rim[bins == b].mean(axis=0) for b in np.unique(bins)])
    seg = np.roll(centres, -1, axis=0) - centres
    seg /= np.linalg.norm(seg, axis=1, keepdims=True) + 1e-12
    cos = np.einsum("ij,ij->i", seg, np.roll(seg, -1, axis=0))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def outline_turns_deg(
    mesh: Mesh, axis: int, step_rad: float, view: tuple[float, float, float, float]
) -> Array:
    """Turning angles (degrees) of the outline of ``mesh`` seen along ``axis``.

    ``axis`` 0 is the front view and 1 the profile. Rays ``step_rad`` apart
    leave the point ``view[:2]`` of the picture plane at the angles
    ``view[2]..view[3]`` from straight up (positive toward the first picture
    axis); each stops at the farthest edge it crosses. Preconditions: axis 0
    or 1, a positive finite step and at least three rays in the window.
    """
    if axis not in (0, 1) or not 0.0 < step_rad < math.inf:
        raise ValueError("outline needs axis 0 or 1 and a positive finite step")
    angles = np.arange(view[2], view[3] + 0.5 * step_rad, step_rad)
    if len(angles) < 3:
        raise ValueError("outline window holds fewer than three rays")
    flat = mesh.vertices[:, [1 - axis, 2]] - np.asarray(view[:2])
    f = mesh.faces
    edges = np.unique(np.sort(np.concatenate([f[:, :2], f[:, 1:]]), axis=1), axis=0)
    p, q = flat[edges[:, 0]], flat[edges[:, 1]]
    rays = np.stack([np.sin(angles), np.cos(angles)], axis=1)
    side_p = p[:, 0][None] * rays[:, 1:] - p[:, 1][None] * rays[:, :1]
    side_q = q[:, 0][None] * rays[:, 1:] - q[:, 1][None] * rays[:, :1]
    t = side_p / np.where(side_p == side_q, 1.0, side_p - side_q)
    radius = np.einsum("rej,rj->re", p[None] + t[..., None] * (q - p)[None], rays)
    radius[(side_p > 0) == (side_q > 0)] = 0.0
    line = radius.max(axis=1)[:, None] * rays
    seg = np.diff(line, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True) + 1e-12
    cos = np.einsum("ij,ij->i", seg[:-1], seg[1:])
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def comb_ridge_vertices(skin: Mesh, part: str, crop: HairCrop | None = None) -> float:
    """Vertices per combed ridge: the wavelength over the longest crown edge."""
    cfg = crop or load_hair_crop()
    v, _, faces = _scalp(skin, cfg, part)
    crown = faces[(v[faces, 2] > cfg.crown_from_z + cfg.crown_band).all(axis=1)]
    edge = v[crown] - v[np.roll(crown, -1, axis=1)]
    return float(cfg.comb_wavelength / np.linalg.norm(edge, axis=2).max())


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
    cfg = crop or load_hair_crop()
    v, n, faces = _scalp(skin, cfg, part)
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
