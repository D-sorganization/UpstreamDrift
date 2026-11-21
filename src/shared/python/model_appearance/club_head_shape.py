"""Rounded head body and blended hosel for the drawn driver (#15150).

Settings: ``club_head_shape_settings.json``. Taubin smoothing rounds the boxy
Tools body with the face plate pinned; the hosel flares into the heel and eases
to the shaft tip radius. Head frame, metres, visual only: the face, sole point,
hosel top, mass and inertia do not move.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from src.shared.python.model_appearance.club_shaft_mesh import profiled_tube
from src.shared.python.model_appearance.geometry import Mesh

Array = NDArray[np.float64]
SETTINGS_PATH = Path(__file__).with_name("club_head_shape_settings.json")
_AXIS_SAMPLES = 256


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class HeadShapeSettings:
    """Validated ``club_head_shape_settings.json`` (``ValueError`` if bad)."""

    shaped_club_types: tuple[str, ...]
    smoothing_iterations: int
    smoothing_lambda: float
    smoothing_mu: float
    face_pin_cos: float
    hosel_radius_m: float
    hosel_top_radius_m: float
    hosel_flare_radius_m: float
    hosel_base_blend_m: float
    hosel_cylinder_length_m: float
    hosel_shoulder_height_m: float
    hosel_profile_steps: int
    hosel_sides: int

    def __post_init__(self) -> None:
        its = self.smoothing_iterations
        if isinstance(its, bool) or not isinstance(its, int) or its < 0:
            raise ValueError("smoothing_iterations must be an integer of at least 0")
        if not 0.0 < self.smoothing_lambda < 1.0:
            raise ValueError("smoothing_lambda must be in (0, 1)")
        if not self.smoothing_mu < -self.smoothing_lambda:
            raise ValueError("smoothing_mu must be below minus smoothing_lambda")
        if not 0.0 < self.face_pin_cos < 1.0:
            raise ValueError("face_pin_cos must be in (0, 1)")
        for name in (f.name for f in fields(self) if f.name.endswith("_m")):
            _positive(getattr(self, name), name)
        top, body, flare = (
            self.hosel_top_radius_m,
            self.hosel_radius_m,
            self.hosel_flare_radius_m,
        )
        if not top <= body <= flare:
            raise ValueError("hosel radii must satisfy top <= body <= flare")
        if self.hosel_profile_steps < 2 or self.hosel_sides < 6:
            raise ValueError("hosel_profile_steps >= 2 and hosel_sides >= 6 required")
        object.__setattr__(self, "shaped_club_types", tuple(self.shaped_club_types))


def load_head_shape_settings(path: Path | None = None) -> HeadShapeSettings:
    """Read the settings file; ``KeyError`` for a missing key, else as the class."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return HeadShapeSettings(*(raw[f.name] for f in fields(HeadShapeSettings)))


# ----------------------------------------------------------------- the body


def _face_pin_vertices(mesh: Mesh, normal: Array, cos_min: float) -> NDArray[np.bool_]:
    """Mask of the vertices of the face-plate triangles (never moved)."""
    tri = mesh.vertices[mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn = cross / np.maximum(np.linalg.norm(cross, axis=1), 1e-18)[:, None]
    x_mid = 0.5 * float(mesh.vertices[:, 0].min() + mesh.vertices[:, 0].max())
    face = (fn @ normal > cos_min) & (tri[:, :, 0].mean(axis=1) > x_mid)
    pinned = np.zeros(len(mesh.vertices), dtype=bool)
    pinned[mesh.faces[face].ravel()] = True
    return pinned


def round_head_body(mesh: Mesh, face_normal: Array, s: HeadShapeSettings) -> Mesh:
    """``mesh`` after Taubin smoothing with the face plate pinned.

    Preconditions: ``face_normal`` is a unit 3-vector. Postcondition: same
    faces (so the winding stays outward and the mesh stays closed); the
    face-plate vertices are bit-identical.
    """
    normal = np.asarray(face_normal, dtype=float)
    if normal.shape != (3,) or abs(float(np.linalg.norm(normal)) - 1.0) > 1e-6:
        raise ValueError("face_normal must be a unit 3-vector")
    if s.smoothing_iterations == 0:
        return mesh
    pinned = _face_pin_vertices(mesh, normal, s.face_pin_cos)
    pairs = np.vstack(
        [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]
    )
    edges = np.unique(np.sort(pairs, axis=1), axis=0)
    count = np.maximum(np.bincount(edges.ravel(), minlength=len(mesh.vertices)), 1.0)
    verts = mesh.vertices.copy()
    free = ~pinned
    for _ in range(s.smoothing_iterations):
        for step in (s.smoothing_lambda, s.smoothing_mu):
            total = np.zeros_like(verts)
            np.add.at(total, edges[:, 0], verts[edges[:, 1]])
            np.add.at(total, edges[:, 1], verts[edges[:, 0]])
            lap = total / count[:, None] - verts
            verts[free] += step * lap[free]
    # Keep the head's height range as built: face-centre picking and the sole
    # plane both read it, and smoothing shrinks the range by under a millimetre.
    lo, hi = float(verts[:, 1].min()), float(verts[:, 1].max())
    top, bottom = float(mesh.vertices[:, 1].max()), float(mesh.vertices[:, 1].min())
    verts[free, 1] = bottom + (verts[free, 1] - lo) * (top - bottom) / (hi - lo)
    return Mesh(verts, mesh.faces)


# ----------------------------------------------------------------- the hosel


def axis_contact_param(
    body: Mesh, sole: Array, direction: Array, t_top: float, reach: float
) -> float:
    """Largest ``t`` in ``[0, t_top]`` where the axis ``sole + t * direction``
    is within ``reach`` of the body: the last height a hosel still touches the
    head (the driver seat puts the axis on the heel edge, clear of the crown).
    Raises ``ValueError`` when it never comes within ``reach``.
    """
    tri = body.vertices[body.faces]
    cloud = np.vstack([body.vertices, tri.mean(axis=1)])
    t = np.linspace(0.0, t_top, _AXIS_SAMPLES)
    near, _ = cKDTree(cloud).query(sole + t[:, None] * direction)
    touching = np.asarray(near) <= reach
    if not touching.any():
        raise ValueError("hosel axis never comes within reach of the head body")
    return float(t[touching][-1])


def hosel_stations(
    t_contact: float, t_top: float, s: HeadShapeSettings
) -> list[tuple[float, float]]:
    """``(t, radius)`` hosel profile along the axis.

    A rounded base inside the heel, a collar of ``hosel_flare_radius_m`` at the
    last head contact, a fairing down to ``hosel_radius_m``, a short cylinder
    and a shoulder to ``hosel_top_radius_m`` under the shaft. Postcondition:
    ``t`` strictly rising, ending at ``t_top`` with the shaft tip radius.
    Raises ``ValueError`` when the span above the contact is too short.
    """
    t_low = max(0.0, t_contact - s.hosel_base_blend_m)
    t_fair = t_top - s.hosel_shoulder_height_m - s.hosel_cylinder_length_m
    if t_fair <= t_contact or t_contact <= t_low:
        raise ValueError("hosel span above the head contact is too short")
    top, body, flare = s.hosel_top_radius_m, s.hosel_radius_m, s.hosel_flare_radius_m
    t_sh = t_top - s.hosel_shoulder_height_m
    legs = (
        (t_low, t_contact, top, flare),
        (t_contact, t_fair, flare, body),
        (t_fair, t_sh, body, body),
        (t_sh, t_top, body, top),
    )
    x = np.linspace(0.0, 1.0, s.hosel_profile_steps + 1)
    w = 0.5 * (1.0 - np.cos(np.pi * x))  # cosine ease: flat slope at both ends
    out: list[tuple[float, float]] = []
    for t0, t1, r0, r1 in legs:
        pts = list(
            zip(
                (t0 + (t1 - t0) * x).tolist(),
                (r0 + (r1 - r0) * w).tolist(),
                strict=True,
            )
        )
        out += pts[1:] if out else pts
    return out


def blended_hosel(
    body: Mesh, sole: Array, direction: Array, t_top: float, s: HeadShapeSettings
) -> Mesh:
    """Closed hosel from inside the heel to ``t_top`` (the shaft end): base
    collar on the head, fairing, cylinder, and a shoulder that meets the shaft
    tip radius."""
    t_contact = axis_contact_param(body, sole, direction, t_top, s.hosel_radius_m)
    stations = hosel_stations(t_contact, t_top, s)
    t0 = stations[0][0]
    span = t_top - t0
    frac = [((t - t0) / span, r) for t, r in stations]
    frac[0], frac[-1] = (0.0, frac[0][1]), (1.0, frac[-1][1])
    return profiled_tube(
        sole + t0 * direction, sole + t_top * direction, frac, s.hosel_sides
    )
