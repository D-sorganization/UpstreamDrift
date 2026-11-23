"""Score lines and face colour for the drawn club head (#15150).

Settings: ``club_face_marking_settings.json`` (default: head as built). The face
normal and centre come from ``club_head_mesh``; the toe axis is the head
frame's ``+z``. Head frame, metres, visual only.
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
_SAMPLE_STEP_M = 0.0015
_N_STEPS = 60  # samples each side of the face centre along the toe axis
_RAY_START_M = 0.03
FACE_NORMAL_COS = 0.9  # a triangle is face within about 26 degrees of the normal
_EDGE_MARGIN_M = 0.004  # line ends keep this far from the face edge
_KEYS = (
    "face_marking",
    "score_line_count",
    "score_line_spacing_m",
    "score_line_width_m",
    "score_line_height_m",
    "centre_clear_width_m",
    "face_rgba",
    "line_rgba",
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
    face_rgba: tuple[float, ...]
    line_rgba: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.face_marking, bool):
            raise ValueError("face_marking must be true or false")
        count = self.score_line_count
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("score_line_count must be an integer of at least 0")
        _positive(self.score_line_spacing_m, "score_line_spacing_m")
        _positive(self.score_line_width_m, "score_line_width_m")
        _positive(self.score_line_height_m, "score_line_height_m")
        if (
            not math.isfinite(self.centre_clear_width_m)
            or self.centre_clear_width_m < 0
        ):
            raise ValueError("centre_clear_width_m must be finite and not negative")
        if self.score_line_width_m >= self.score_line_spacing_m:
            raise ValueError("score_line_width_m must be less than the spacing")
        for name in ("face_rgba", "line_rgba"):
            object.__setattr__(self, name, _rgba(getattr(self, name), name))


def load_face_marking_settings(path: Path | None = None) -> FaceMarkingSettings:
    """Read the settings file; ``KeyError`` for a missing key, else as the class."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return FaceMarkingSettings(*(raw[key] for key in _KEYS))


# ------------------------------------------------------------------ helpers


def _compact(mesh: Mesh) -> Mesh:
    used, inverse = np.unique(mesh.faces, return_inverse=True)
    faces = np.asarray(inverse, dtype=np.int64).reshape(mesh.faces.shape)
    return Mesh(mesh.vertices[used], faces)


# ------------------------------------------------------------- face and lines


@dataclass(frozen=True)
class FaceFrame:
    """Face centre and unit axes in the head frame (from the head geometry)."""

    centre: Array
    normal: Array
    toe: Array
    up: Array


def face_frame(head_body: Mesh) -> FaceFrame:
    """Measured normal and centre of the head, toe ``+z`` made orthogonal."""
    normal = measured_face_normal(head_body)
    toe = np.array([0.0, 0.0, 1.0])
    toe = toe - float(toe @ normal) * normal
    toe /= np.linalg.norm(toe)
    return FaceFrame(face_centre(head_body), normal, toe, np.cross(toe, normal))


def face_triangle_mask(mesh: Mesh, frame: FaceFrame, cos_min: float) -> np.ndarray:
    """Boolean mask of the triangles that make up the face."""
    tri = mesh.vertices[mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn = cross / np.maximum(np.linalg.norm(cross, axis=1), 1e-18)[:, None]
    centroid = tri.mean(axis=1)
    ahead = (centroid - frame.centre) @ frame.normal > -0.02
    return np.asarray((fn @ frame.normal > cos_min) & ahead)


def face_surface_triangles(mesh: Mesh) -> Array:
    """Return the rendered face patch, not a contact or first-root verdict."""
    mask = face_triangle_mask(mesh, face_frame(mesh), FACE_NORMAL_COS)
    triangles = np.asarray(mesh.vertices[mesh.faces[mask]], dtype=float).copy()
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError("face patch must contain triangles")
    area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if (
        not np.isfinite(triangles).all()
        or not np.isfinite(area).all()
        or np.any(area <= 0)
    ):
        raise ValueError("face patch must contain finite nondegenerate triangles")
    return triangles


def _ray_heights(tri: Array, origins: Array, direction: Array) -> Array:
    """Distance to the nearest triangle hit along ``direction``, inf if none."""
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


def _line_strips(
    face_tri: Array, frame: FaceFrame, s: FaceMarkingSettings
) -> list[tuple[Array, Array]]:
    """``(low edge, high edge)`` lifted surface points of each line segment,
    each edge cast on the surface so ribbons follow the face curvature."""
    count = s.score_line_count
    t = np.arange(-_N_STEPS, _N_STEPS + 1) * _SAMPLE_STEP_M
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
        lo, hi = t[ok].min() + _EDGE_MARGIN_M, t[ok].max() - _EDGE_MARGIN_M
        for side in ((t <= -half_clear) & (t >= lo), (t >= half_clear) & (t <= hi)):
            keep = side & ok
            if keep.sum() >= 2:
                strips.append((edges[0][0][keep], edges[1][0][keep]))
    return strips


def _ribbon(low: Array, high: Array) -> Mesh:
    n = len(low)
    verts = np.vstack([low, high])
    i = np.arange(n - 1)
    faces = [[i, n + i, i + 1], [i + 1, n + i, n + i + 1]]
    return Mesh(verts, np.vstack([np.column_stack(f) for f in faces]).astype(np.int64))


def score_line_mesh(
    head_body: Mesh, frame: FaceFrame, settings: FaceMarkingSettings
) -> Mesh | None:
    """Ribbons of the score lines, centred up the face, split by the clear
    zone, ``score_line_height_m`` above the surface; ``None`` if no line lands."""
    mask = face_triangle_mask(head_body, frame, FACE_NORMAL_COS)
    tri = head_body.vertices[head_body.faces[mask]]
    strips = _line_strips(tri, frame, settings)
    if not strips:
        return None
    return concat_meshes(*(_ribbon(low, high) for low, high in strips))


# ----------------------------------------------------------------- the parts


def marked_head_parts(
    head: ClubHeadMesh, settings: FaceMarkingSettings
) -> dict[str, Mesh]:
    """Head-frame ``head`` (body less face, plus hosel), ``head_face`` and,
    when any line lands on the face, ``face_lines``."""
    body = head.head_mesh
    hosel = Mesh(
        head.mesh.vertices[len(head.head_mesh.vertices) :],
        head.mesh.faces[len(head.head_mesh.faces) :] - len(head.head_mesh.vertices),
    )
    frame = face_frame(head.head_mesh)
    mask = face_triangle_mask(body, frame, FACE_NORMAL_COS)
    shell = _compact(Mesh(body.vertices, body.faces[~mask]))
    parts = {
        "head": concat_meshes(shell, hosel),
        "head_face": _compact(Mesh(body.vertices, body.faces[mask])),
    }
    lines = score_line_mesh(body, frame, settings)
    if lines is not None:
        parts["face_lines"] = lines
    return parts
