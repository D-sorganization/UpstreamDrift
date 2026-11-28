"""Eyes, brows, lids and mouth placed from the CC0 skin mesh landmarks (#15276).

The MakeHuman head carries sculpted lids, nose, lips and ears but no eyeballs
or colour breaks, so it reads as a blank mannequin. These parts are built in
head-length units from :class:`FaceLandmarks`; brows and mouth are conformed
to the local skin surface, and the eyeballs sit behind the open lids. Visual
only, like the rest of the head.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from src.shared.python.model_appearance.geometry import Array, Mesh, ellipsoid_mesh

_X, _Y = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
SETTINGS_PATH = (
    Path(__file__).resolve().parents[4] / "assets/head_cc0/face_settings.json"
)
_RING = np.array([[0.0, 0.0], [1, 0], [-1, 0], [0, 1], [0, -1]])  # cover stencil


@lru_cache(maxsize=1)
def _settings() -> dict[str, Any]:
    """Read and cache the authored CC0 facial landmark settings."""
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def _frozen(value: Any) -> Any:
    """Convert mutable JSON lists into tuples for frozen landmark records."""
    return tuple(_frozen(v) for v in value) if isinstance(value, list) else value


def face_settings(section: str) -> dict[str, Any]:
    """One section (``face``, ``hair`` or ``checks``) of ``face_settings.json``."""
    data = _settings()
    if not isinstance(data.get(section), dict):
        raise ValueError(f"{SETTINGS_PATH.name} has no section {section!r}")
    return {key: _frozen(value) for key, value in data[section].items()}


@dataclass(frozen=True)
class FaceLandmarks:
    """Feature placement in head lengths (x forward, y left, z up).

    The sculpted lids of the CC0 head are open over an empty socket, so the
    eye is a ball seated behind them with the iris and the pupil each proud of
    the layer under it by its own lift; a lash line follows the upper lid.
    The values in use are those of ``face_settings.json``
    (:func:`load_face_landmarks`); a non-finite number, a size that is not
    positive or a negative lift raises ``ValueError``.
    """

    eye_y: float = 0.118  # half the pupillary distance, from the sculpted slit
    eye_z: float = 0.532
    eye_radius: float = 0.048  # the eyeball
    eye_inset: float = 0.006  # eyeball front behind the lid margins
    eye_cover: float = 0.016  # the lid margins are the skin's front this near
    strip_cover: float = 0.003  # the same reach for the lid line and the brow
    iris_half: tuple[float, float, float] = (0.006, 0.020, 0.020)
    iris_drop: float = 0.003  # iris centre below the eye centre
    iris_lift: float = 0.008
    pupil_half: tuple[float, float, float] = (0.005, 0.0085, 0.0085)
    pupil_lift: float = 0.0115
    lid_rise: float = 0.022  # upper lid line above the eye centre
    lid_half: tuple[float, float, float] = (0.005, 0.050, 0.005)
    lid_arch: float = 0.0  # ends drop by this fraction of the half length
    lid_lift: float = 0.0085
    brow_rise: float = 0.070
    brow_half: tuple[float, float, float] = (0.010, 0.072, 0.012)
    brow_tilt: float = 0.16  # outer end rises by this fraction of its length
    brow_arch: float = 0.0
    brow_taper: float = 0.0  # share of its height the outer end loses
    brow_lift: float = 0.004
    mouth_z: float = 0.276
    mouth_half: tuple[float, float, float] = (0.008, 0.075, 0.006)
    mouth_lift: float = 0.003

    def __post_init__(self) -> None:
        """Reject non-finite sizes, non-positive dimensions, and negative lifts."""
        for item in fields(self):
            value = np.atleast_1d(np.asarray(getattr(self, item.name), dtype=float))
            sized = item.name.endswith(("_half", "_cover", "_radius"))
            if not np.isfinite(value).all() or (sized and (value <= 0.0).any()):
                raise ValueError(f"face setting {item.name} is not a valid size")
            if item.name.endswith(("_lift", "_inset")) and (value < 0.0).any():
                raise ValueError(f"face setting {item.name} must not be negative")


def load_face_landmarks() -> FaceLandmarks:
    """The face of ``face_settings.json`` (its ``face`` section over the defaults)."""
    try:
        return FaceLandmarks(**face_settings("face"))
    except TypeError as unknown:
        raise ValueError(f"unknown face setting: {unknown}") from unknown


def barycentric(triangles: Array, point: Array) -> Array:
    """Barycentric weights of a 2-D ``point`` in each of ``(n, 3, 2)`` triangles.

    A degenerate triangle gets negative weights, so it never contains a point.
    """
    a, b, c = (triangles[:, k] - point for k in range(3))
    areas = np.stack(
        [
            b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0],
            c[:, 0] * a[:, 1] - c[:, 1] * a[:, 0],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ],
        axis=1,
    )
    total = areas.sum(axis=1, keepdims=True)
    return np.where(np.abs(total) > 1e-14, areas / np.where(total, total, 1.0), -1.0)


def front_depth(mesh: Mesh, points_yz: Array) -> Array:
    """Front-most x of ``mesh`` on the x lines through ``points_yz`` (nan off it)."""
    points = np.asarray(points_yz, dtype=float).reshape(-1, 2)
    tri = mesh.vertices[mesh.faces]
    low, high = tri[:, :, 1:3].min(axis=1), tri[:, :, 1:3].max(axis=1)
    near = (high >= points.min(axis=0)).all(axis=1) & (low <= points.max(axis=0)).all(1)
    tri = tri[near]
    out = np.full(len(points), np.nan)
    for i, point in enumerate(points):
        weights = barycentric(tri[:, :, 1:3], point)
        hit = (weights >= 0.0).all(axis=1)
        if hit.any():
            out[i] = (weights[hit] * tri[hit, :, 0]).sum(axis=1).max()
    return out


def _on_cover(skin: Mesh, mesh: Mesh, lift: float, radius: float) -> Mesh:
    """``mesh`` (built about x = 0) laid ``lift`` proud of the skin's cover.

    The cover at a point is the skin's highest front within ``radius``, so a
    part never sinks under a sculpted lid or ridge next to it.
    """
    yz = mesh.vertices[:, 1:3]
    cover = np.nanmax([front_depth(skin, yz + radius * o) for o in _RING], axis=0)
    if not np.isfinite(cover).all():
        raise ValueError("a face part reaches beyond the skin of the face")
    out = mesh.vertices.copy()
    out[:, 0] += cover + lift
    return Mesh(out, mesh.faces)


def _strip(
    centre_yz: tuple[float, float],
    half: tuple[float, float, float],
    sign: float,
    shape: tuple[float, float, float],
) -> Mesh:
    """Strip along y about x = 0; ``shape`` is ``(tilt, arch, taper)``.

    The outer end rises by ``tilt`` and both ends drop by ``arch`` (fractions
    of the half length); the outer end loses ``taper`` of its height.
    """
    tilt, arch, taper = shape
    mesh = _blob(np.array([0.0, *centre_yz]), half, _Y)
    v = mesh.vertices.copy()
    s = sign * (v[:, 1] - centre_yz[0]) / half[1]
    thin = 1.0 - taper * np.clip(0.5 * (s + 1.0), 0.0, 1.0)
    v[:, 2] = centre_yz[1] + (v[:, 2] - centre_yz[1]) * thin
    v[:, 2] += half[1] * (tilt * s - arch * s * s)
    return Mesh(v, mesh.faces)


class FaceSurface:
    """Outer skin depth x(y, z) of the face front, linearly interpolated.

    The head mesh also holds eye-socket and lip interiors behind the visible
    skin, so only the outermost vertex per ``envelope_cell`` is kept; a small
    stencil average then removes the jitter of the 4 mm triangles.
    """

    def __init__(
        self,
        skin: Mesh,
        envelope_cell: float = 0.012,
        blur: float = 0.006,
        min_x: float = 0.2,
    ) -> None:
        """Build interpolated front-surface depth from the outer skin vertices."""
        front = skin.vertices[skin.vertices[:, 0] > min_x]
        outer: dict[tuple[int, int], Array] = {}
        for p in front:
            cell = (round(p[1] / envelope_cell), round(p[2] / envelope_cell))
            if cell not in outer or p[0] > outer[cell][0]:
                outer[cell] = p
        pts = np.array(list(outer.values()))
        self._linear = LinearNDInterpolator(pts[:, 1:3], pts[:, 0])
        self._nearest = NearestNDInterpolator(pts[:, 1:3], pts[:, 0])
        self._stencil = blur * np.array(
            [[0.0, 0.0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float
        )

    def depth(self, y: float, z: float, scale: float = 1.0) -> float:
        """Skin depth x at (y, z), blurred over a five-point stencil.

        ``scale`` widens the stencil for features (the eyes) that must ignore
        the sculpted detail under them.
        """
        q = scale * self._stencil + np.array([y, z])
        vals = self._linear(q)
        bad = np.isnan(vals)
        vals[bad] = self._nearest(q[bad])
        return float(vals.mean())

    def conform(
        self, mesh: Mesh, centre: Array, lift: float, scale: float = 1.0
    ) -> Mesh:
        """Shift each vertex in x by how far the skin there is from the centre."""
        base = self.depth(centre[1], centre[2], scale)
        out = mesh.vertices.copy()
        out[:, 0] += [self.depth(p[1], p[2], scale) - base for p in mesh.vertices]
        out[:, 0] += lift
        return Mesh(out, mesh.faces)


def _blob(centre: Array, half_xyz: tuple[float, float, float], axis: Array) -> Mesh:
    """Ellipsoid with ``(x, y, z)`` half sizes whose long axis is x or y-ish."""
    hx, hy, hz = half_xyz
    if abs(axis[0]) > 0.9:  # along x: sizes (x, y, z), width runs along y
        return ellipsoid_mesh(
            centre, np.array([hx, hy, hz]), axis, rings=6, sides=12, width_hint=_Y
        )
    return ellipsoid_mesh(  # along y: sizes (y, x, z), width runs along x
        centre, np.array([hy, hx, hz]), axis, rings=6, sides=12, width_hint=_X
    )


def moustache_parts(skin: Mesh) -> list[tuple[str, Mesh, str]]:
    """Two closed tapered hair strips conformed above the existing upper lip."""
    out = []
    for side, sign in (("l", 1.0), ("r", -1.0)):
        mesh = _strip(
            (sign * 0.04, 0.315), (0.002, 0.043, 0.008), sign, (-0.06, 0.12, 0.3)
        )
        out.append((f"moustache_{side}", _on_cover(skin, mesh, 0.001, 0.002), "brow"))
    return out


def short_beard_parts(skin: Mesh) -> list[tuple[str, Mesh, str]]:
    """Create a close-cropped stubble shell, with a clear lip opening."""
    return _beard_parts(skin, upper_extension=0.0, shell_offset=0.003, role="beard")


def dense_short_beard_parts(skin: Mesh) -> list[tuple[str, Mesh, str]]:
    """Create a fuller short-beard shell that still leaves the lips open."""
    return _beard_parts(
        skin, upper_extension=0.035, shell_offset=0.004, role="dense_beard"
    )


def _beard_parts(
    skin: Mesh, *, upper_extension: float, shell_offset: float, role: str
) -> list[tuple[str, Mesh, str]]:
    """Build the facial-hair cover with a clear lip opening and fixed topology."""
    from src.shared.python.model_appearance.head_hair import (
        HairCrop,
        _clip_above,
        _split,
        _vertex_normals,
        closed_shell,
    )

    v, normals, faces = skin.vertices, _vertex_normals(skin), skin.faces
    faces = faces[((v[faces, 2] < 0.48) & (v[faces, 0] > 0.08)).any(axis=1)]
    for _ in range(2):
        v, normals, faces = _split(
            v, normals, faces, HairCrop(refine_edge=0.012, refine_curve=0)
        )

    def level(points: Array) -> Array:
        """Signed distance proxy for cheek, chin and elliptical lip boundaries."""
        x, y, z = np.moveaxis(points, -1, 0)
        side = np.clip((np.abs(y) - 0.09) / 0.14, 0.0, 1.0)
        upper = 0.295 + upper_extension + 0.085 * side * side * (3 - 2 * side)
        outer = np.clip((np.abs(y) - 0.20) / 0.12, 0.0, 1.0)
        front = 0.13 + 0.10 * outer * outer * (3 - 2 * outer)
        lip = 0.05 * (np.hypot(y / 0.09, (z - 0.27) / 0.045) - 1)
        return np.minimum.reduce((x - front, z - 0.075, upper - z, lip))

    v, normals, faces = _clip_above(
        v, normals, faces, level(v), (lambda p: float(level(p)), 8)
    )
    mesh = closed_shell(v, normals, faces, np.full(len(v), shell_offset), 0.0004, 0.001)
    return [
        ("beard_shell", mesh, role),
        *[(name, part, role) for name, part, _ in moustache_parts(skin)],
    ]


def face_parts(
    skin: Mesh, marks: FaceLandmarks | None = None
) -> list[tuple[str, Mesh, str]]:
    """``(name, mesh, material role)`` of eyes, lids, brows and mouth.

    ``skin`` is the unit head-length skin mesh. Postcondition: one entry per
    part and side, left (y > 0) and right; each eye is an eyeball seated behind
    the lid margins with iris and pupil proud of it (``face_settings.json``).
    """
    m = marks or load_face_landmarks()
    surf = FaceSurface(skin)
    front = Mesh(skin.vertices, skin.faces[(skin.vertices[skin.faces, 0] > 0).all(1)])

    def decal(centre_yz, half, axis, lift):
        """Place one facial detail against the interpolated skin surface."""
        c = np.array([surf.depth(centre_yz[0], centre_yz[1]), *centre_yz])
        return surf.conform(_blob(c, half, axis), c, lift)

    out: list[tuple[str, Mesh, str]] = []
    for side, sign in (("l", 1.0), ("r", -1.0)):
        y, z = sign * m.eye_y, m.eye_z
        lids = front_depth(front, np.array([y, z]) + m.eye_cover * _RING)
        if not np.isfinite(lids).any():
            raise ValueError("the eye centre is off the skin of the face")
        apex = float(np.nanmax(lids)) - m.eye_inset  # front of the eyeball
        ball = ellipsoid_mesh(
            np.array([apex - m.eye_radius, y, z]),
            np.full(3, m.eye_radius),
            _X,
            rings=6,
            sides=12,
            width_hint=_Y,
        )
        out.append((f"eye_{side}", ball, "eye_white"))
        for name, half, lift in (
            ("iris", m.iris_half, m.iris_lift),
            ("pupil", m.pupil_half, m.pupil_lift),
        ):
            centre = np.array([apex + lift - half[0], y, z - m.iris_drop])
            out.append((f"{name}_{side}", _blob(centre, half, _X), name))
        lid = _strip((y, z + m.lid_rise), m.lid_half, sign, (0.0, m.lid_arch, 0.0))
        out.append(
            (f"lid_{side}", _on_cover(front, lid, m.lid_lift, m.strip_cover), "lid")
        )
        shape = (m.brow_tilt, m.brow_arch, m.brow_taper)
        brow = _strip((y, z + m.brow_rise), m.brow_half, sign, shape)
        brow = _on_cover(front, brow, m.brow_lift, m.strip_cover)
        out.append((f"brow_{side}", brow, "brow"))
    mouth = decal((0.0, m.mouth_z), m.mouth_half, _Y, m.mouth_lift)
    out.append(("mouth", mouth, "lip_seam"))
    return out
