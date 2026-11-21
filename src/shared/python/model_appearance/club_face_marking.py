"""Score lines, face colour and smoothing for the drawn club head (#15150).

``club_face_marking_settings.json`` beside this module names the choices; the
defaults leave the head exactly as :mod:`club_head_mesh` builds it. Nothing
here builds a head: the face normal and centre come from
``club_head_mesh.measured_face_normal`` / ``face_centre`` and the toe axis is
the head frame's ``+z`` documented there. Everything is in the head frame
(metres); the caller seats the parts like the head. Visual only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.club_head_mesh import (
    ClubHeadMesh,
    face_centre,
    measured_face_normal,
)
from src.shared.python.model_appearance.geometry import Mesh, concat_meshes

Array = NDArray[np.float64]
SETTINGS_PATH = Path(__file__).with_name("club_face_marking_settings.json")
MAX_SUBDIVISIONS = 3
_SAMPLE_STEP_M = 0.0015
_N_STEPS = 60  # samples each side of the face centre along the toe axis
_RAY_START_M = 0.03
_KEYS = (
    "face_marking",
    "score_line_count",
    "score_line_spacing_m",
    "score_line_width_m",
    "score_line_height_m",
    "centre_clear_width_m",
    "edge_margin_m",
    "face_rgba",
    "line_rgba",
    "face_normal_cos",
    "subdivisions",
    "crease_angle_deg",
    "max_triangles",
)


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def _rgba(value: object, name: str) -> tuple[float, ...]:
    colour = tuple(float(c) for c in value)  # type: ignore[attr-defined]
    if len(colour) != 4 or not all(0.0 <= c <= 1.0 for c in colour):
        raise ValueError(f"{name} must be four values in [0, 1]")
    return colour


@dataclass(frozen=True)
class FaceMarkingSettings:
    """Validated ``club_face_marking_settings.json`` (``ValueError`` if bad)."""

    face_marking: bool
    score_line_count: int
    score_line_spacing_m: float
    score_line_width_m: float
    score_line_height_m: float
    centre_clear_width_m: float
    edge_margin_m: float
    face_rgba: tuple[float, ...]
    line_rgba: tuple[float, ...]
    face_normal_cos: float
    subdivisions: int
    crease_angle_deg: float
    max_triangles: int

    def __post_init__(self) -> None:
        if not isinstance(self.face_marking, bool):
            raise ValueError("face_marking must be true or false")
        for name in ("score_line_count", "subdivisions", "max_triangles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer of at least 0")
        if self.subdivisions > MAX_SUBDIVISIONS:
            raise ValueError(f"subdivisions must not exceed {MAX_SUBDIVISIONS}")
        for name in ("score_line_spacing_m", "score_line_width_m", "crease_angle_deg"):
            _positive(getattr(self, name), name)
        _positive(self.score_line_height_m, "score_line_height_m")
        for name in ("centre_clear_width_m", "edge_margin_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be finite and not negative")
        if self.score_line_width_m >= self.score_line_spacing_m:
            raise ValueError("score_line_width_m must be less than the spacing")
        if not 0.0 < self.face_normal_cos < 1.0:
            raise ValueError("face_normal_cos must be between 0 and 1")
        object.__setattr__(self, "face_rgba", _rgba(self.face_rgba, "face_rgba"))
        object.__setattr__(self, "line_rgba", _rgba(self.line_rgba, "line_rgba"))

    @property
    def active(self) -> bool:
        """True when the drawn head differs from the one as built."""
        return self.face_marking or self.subdivisions > 0


def load_face_marking_settings(path: Path | None = None) -> FaceMarkingSettings:
    """Read the settings file; ``KeyError`` for a missing key, else as the class."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return FaceMarkingSettings(*(raw[key] for key in _KEYS))


# ------------------------------------------------------------------ smoothing


def _face_normals(mesh: Mesh) -> tuple[Array, Array]:
    tri = mesh.vertices[mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    return cross / np.maximum(area[:, None], 1e-18), 0.5 * area


def _edge_end_normals(
    mesh: Mesh, ends: Array, edge_dir: Array, fn: Array, area: Array, cos_min: float
) -> Array:
    """Per edge end: area-weighted mean normal of the faces at the vertex that
    lie within the crease angle of the edge's own normal ``edge_dir``."""
    flat = mesh.faces.ravel()
    order = np.argsort(flat, kind="stable")
    degree = np.bincount(flat, minlength=len(mesh.vertices))
    start = np.concatenate([[0], np.cumsum(degree)[:-1]])
    count = degree[ends]
    owner = np.repeat(np.arange(len(ends)), count)
    offset = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    face = order[start[ends][owner] + offset] // 3
    keep = np.einsum("ij,ij->i", fn[face], edge_dir[owner]) > cos_min
    out = np.zeros((len(ends), 3))
    np.add.at(out, owner[keep], fn[face[keep]] * area[face[keep], None])
    norm = np.linalg.norm(out, axis=1)
    return np.where(
        norm[:, None] > 1e-12, out / np.maximum(norm, 1e-12)[:, None], edge_dir
    )


def _subdivide_once(mesh: Mesh, crease_deg: float) -> Mesh:
    verts, faces = mesh.vertices, mesh.faces
    fn, area = _face_normals(mesh)
    half = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    key = np.sort(half, axis=1)
    edges, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    face_of = np.repeat(np.arange(len(faces)), 3)
    summed = np.zeros((len(edges), 3))
    np.add.at(summed, inverse, fn[face_of])
    length = np.linalg.norm(summed, axis=1)
    edge_dir = summed / np.maximum(length, 1e-12)[:, None]
    cos_c = math.cos(math.radians(crease_deg))
    smooth = (length**2 - 2.0) / 2.0 > cos_c  # dot of the two face normals
    ends = np.concatenate([edges[:, 0], edges[:, 1]])
    normals = _edge_end_normals(
        mesh, ends, np.vstack([edge_dir, edge_dir]), fn, area, cos_c
    )
    na, nb = normals[: len(edges)], normals[len(edges) :]
    va, vb = verts[edges[:, 0]], verts[edges[:, 1]]
    mid = 0.5 * (va + vb)
    pa = mid - np.einsum("ij,ij->i", mid - va, na)[:, None] * na
    pb = mid - np.einsum("ij,ij->i", mid - vb, nb)[:, None] * nb
    new = np.where(smooth[:, None], 0.5 * (pa + pb), mid)
    m = (inverse + len(verts)).reshape(-1, 3)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    tris = np.vstack(
        [
            np.column_stack([a, m[:, 0], m[:, 2]]),
            np.column_stack([b, m[:, 1], m[:, 0]]),
            np.column_stack([c, m[:, 2], m[:, 1]]),
            m[:, [0, 1, 2]],
        ]
    )
    return Mesh(np.vstack([verts, new]), tris.astype(np.int64))


def _compact(mesh: Mesh) -> Mesh:
    """``mesh`` without the vertices no face uses."""
    used, inverse = np.unique(mesh.faces, return_inverse=True)
    faces = np.asarray(inverse, dtype=np.int64).reshape(mesh.faces.shape)
    return Mesh(mesh.vertices[used], faces)


def smooth_head_body(mesh: Mesh, settings: FaceMarkingSettings) -> Mesh:
    """``mesh`` after up to ``settings.subdivisions`` shape-keeping passes.

    A pass that would push the triangle count past ``max_triangles`` is not
    run; the input is never reduced. Original vertices keep their positions.
    """
    out = mesh
    for _ in range(settings.subdivisions):
        if 4 * len(out.faces) > settings.max_triangles:
            break
        out = _subdivide_once(out, settings.crease_angle_deg)
    return out


# ------------------------------------------------------------- face and lines


@dataclass(frozen=True)
class FaceFrame:
    """Face centre and unit axes in the head frame (from the head geometry)."""

    centre: Array
    normal: Array
    toe: Array
    up: Array


def face_frame(head_body: Mesh) -> FaceFrame:
    """Face frame of a head-frame body mesh: the existing measured normal and
    centre, the head frame's ``+z`` toe axis made orthogonal to the normal."""
    normal = measured_face_normal(head_body)
    toe = np.array([0.0, 0.0, 1.0])
    toe = toe - float(toe @ normal) * normal
    toe /= np.linalg.norm(toe)
    return FaceFrame(face_centre(head_body), normal, toe, np.cross(toe, normal))


def face_triangle_mask(mesh: Mesh, frame: FaceFrame, cos_min: float) -> np.ndarray:
    """Boolean mask of the triangles that make up the face."""
    fn, _ = _face_normals(mesh)
    centroid = mesh.vertices[mesh.faces].mean(axis=1)
    ahead = (centroid - frame.centre) @ frame.normal > -0.02
    return np.asarray((fn @ frame.normal > cos_min) & ahead)


def _ray_heights(tri: Array, origins: Array, direction: Array) -> Array:
    """Distance along ``direction`` to the nearest triangle hit, NaN for a miss
    (Moeller-Trumbore, all rays against all triangles)."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    safe = np.where(np.abs(det) < 1e-14, np.nan, det)
    t_vec = origins[:, None, :] - tri[None, :, 0, :]
    u = np.einsum("rkj,kj->rk", t_vec, p) / safe
    q = np.cross(t_vec, e1[None, :, :])
    v = (q @ direction) / safe
    dist = np.einsum("kj,rkj->rk", e2, q) / safe
    hit = (u >= 0) & (v >= 0) & (u + v <= 1) & (dist > 0)
    return np.asarray(np.where(hit, dist, np.inf).min(axis=1))


def _toe_samples() -> Array:
    return np.arange(-_N_STEPS, _N_STEPS + 1) * _SAMPLE_STEP_M


def _line_strips(
    face_tri: Array, frame: FaceFrame, s: FaceMarkingSettings
) -> list[tuple[Array, Array]]:
    """``(low edge, high edge)`` surface points, lifted, of each line segment.

    Both edges of a ribbon are cast on the surface separately so the ribbon
    follows the face curvature across its width as well as along it.
    """
    count, t = s.score_line_count, _toe_samples()
    heights = (np.arange(count) - 0.5 * (count - 1)) * s.score_line_spacing_m
    half_clear, half_w = 0.5 * s.centre_clear_width_m, 0.5 * s.score_line_width_m
    strips: list[tuple[Array, Array]] = []
    for h in heights:
        edges = []
        for row in (h - half_w, h + half_w):
            origin = (
                frame.centre
                + t[:, None] * frame.toe
                + row * frame.up
                + _RAY_START_M * frame.normal
            )
            reach = _ray_heights(face_tri, origin, -frame.normal)
            lift = s.score_line_height_m - reach[:, None]
            edges.append((origin + lift * frame.normal, np.isfinite(reach)))
        ok = edges[0][1] & edges[1][1]
        if not ok.any():
            continue
        lo, hi = t[ok].min() + s.edge_margin_m, t[ok].max() - s.edge_margin_m
        for side in ((t <= -half_clear) & (t >= lo), (t >= half_clear) & (t <= hi)):
            keep = side & ok
            if keep.sum() >= 2:
                strips.append((edges[0][0][keep], edges[1][0][keep]))
    return strips


def _ribbon(low: Array, high: Array) -> Mesh:
    n = len(low)
    verts = np.vstack([low, high])
    i = np.arange(n - 1)
    faces = np.vstack(
        [
            np.column_stack([i, n + i, i + 1]),
            np.column_stack([i + 1, n + i, n + i + 1]),
        ]
    )
    return Mesh(verts, faces.astype(np.int64))


def score_line_mesh(
    head_body: Mesh, frame: FaceFrame, settings: FaceMarkingSettings
) -> Mesh | None:
    """Ribbons for the horizontal score lines on the face of ``head_body``.

    Lines are centred up the face about its centre, split by the clear centre
    zone and lie ``score_line_height_m`` above the head surface (found by rays
    along the face normal). ``None`` for zero lines or when no line lands on
    the face. Postcondition: ribbons face along the face normal.
    """
    if settings.score_line_count == 0:
        return None
    mask = face_triangle_mask(head_body, frame, settings.face_normal_cos)
    tri = head_body.vertices[head_body.faces[mask]]
    strips = _line_strips(tri, frame, settings)
    if not strips:
        return None
    return concat_meshes(*(_ribbon(low, high) for low, high in strips))


# ----------------------------------------------------------------- the parts


def marked_head_parts(
    head: ClubHeadMesh, settings: FaceMarkingSettings
) -> dict[str, Mesh]:
    """Head-frame ``head`` (body without face, plus hosel), and with face
    marking ``head_face`` and ``face_lines`` parts, per ``settings``.

    The body is smoothed first; face triangles are those of the smoothed body.
    The hosel is the one :func:`load_club_head` built.
    """
    body = smooth_head_body(head.head_mesh, settings)
    hosel = Mesh(
        head.mesh.vertices[len(head.head_mesh.vertices) :],
        head.mesh.faces[len(head.head_mesh.faces) :] - len(head.head_mesh.vertices),
    )
    if not settings.face_marking:
        return {"head": concat_meshes(body, hosel)}
    frame = face_frame(head.head_mesh)
    mask = face_triangle_mask(body, frame, settings.face_normal_cos)
    shell = _compact(Mesh(body.vertices, body.faces[~mask]))
    parts = {
        "head": concat_meshes(shell, hosel),
        "head_face": _compact(Mesh(body.vertices, body.faces[mask])),
    }
    lines = score_line_mesh(body, frame, settings)
    if lines is not None:
        parts["face_lines"] = lines
    return parts
