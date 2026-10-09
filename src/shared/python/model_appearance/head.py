"""Parametric visual head and neck with a readable face (pure numpy).

The head is built once in a canonical frame (origin at the cervicale, x
forward, y left, z up) and sized from the de Leva head length, so every engine
draws the same shape. A skull with a tapered jaw, eyes with irises, brows, a
nose, a mouth, ears and a neck are separate parts so each can take its own
material; hair or a cap with a visor is optional. The geometry is procedural
(a deformed ellipsoid grid), so there is no third-party mesh and nothing to
license beyond this repository (CC0-1.0 like ``assets/body_part_shapes``).

Everything here is visual only: no mass, inertia, collision or degree of
freedom. :func:`head_override_rotation` defines the visual-only orientation
channel (yaw, pitch, roll) that a gaze controller drives.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import (
    Array,
    LoftOptions,
    Mesh,
    closed_grid_mesh,
    ellipsoid_mesh,
    lofted_segment,
)
from src.shared.python.model_appearance.schema import (
    AppearanceDocument,
    HeadSettings,
)

DEFAULT_HEAD_LENGTH_M = 0.2429  # de Leva (1996) male head length, cervicale to vertex
_RINGS, _SIDES = 26, 40
_SKULL_CENTRE_Z, _SKULL_HALF = 0.58, (0.40, 0.31, 0.42)  # fractions of head length
_AXIS = {"x": 0, "y": 1, "z": 2}
FACE_PARTS = ("eye_l", "eye_r", "iris_l", "iris_r", "brow_l", "brow_r", "nose", "mouth")


@dataclass(frozen=True)
class HeadPart:
    """One mesh of the head in the head frame, with its material role."""

    name: str
    mesh: Mesh
    material: str  # a library material name, or "skin" / "headwear"


@dataclass(frozen=True)
class HeadAnchor:
    """Where the head attaches: a spec body and the neck point inside it."""

    body: str
    origin_m: tuple[float, float, float]
    length_m: float
    source: str  # "head_body" (physics neck) or "torso" (no head body in spec)


def axis_vector(axis: str) -> Array:
    """Unit vector of a signed axis string such as ``"+x"`` or ``"-z"``."""
    if not isinstance(axis, str) or len(axis) != 2 or axis[1] not in _AXIS:
        raise ValueError(f"Axis must look like '+x' or '-z', got {axis!r}")
    if axis[0] not in "+-":
        raise ValueError(f"Axis must look like '+x' or '-z', got {axis!r}")
    v = np.zeros(3)
    v[_AXIS[axis[1]]] = 1.0 if axis[0] == "+" else -1.0
    return v


def canonical_to_body(forward_axis: str, up_axis: str) -> Array:
    """Rotation taking the canonical head frame onto the body frame's axes."""
    fwd, up = axis_vector(forward_axis), axis_vector(up_axis)
    if abs(float(fwd @ up)) > 0.5:
        raise ValueError("forward_axis and up_axis must be different axes")
    return np.column_stack([fwd, np.cross(up, fwd), up])


def head_override_rotation(yaw_rad: float, pitch_rad: float, roll_rad: float) -> Array:
    """Rotation of the head frame for a yaw/pitch/roll gaze sample.

    Intrinsic yaw about +z (left turn positive), then pitch about the head's
    left axis (positive looks up), then roll about the forward axis (positive
    tilts the head toward the right shoulder). Preconditions: finite angles.
    Postcondition: a proper rotation; zero angles give the identity.
    """
    angles = np.array([yaw_rad, pitch_rad, roll_rad], dtype=float)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("yaw, pitch and roll must be finite numbers")
    cy, sy = math.cos(angles[0]), math.sin(angles[0])
    cp, sp = math.cos(angles[1]), math.sin(angles[1])
    cr, sr = math.cos(angles[2]), math.sin(angles[2])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])  # +pitch = up
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def resolve_head_anchor(spec: Mapping[str, Any]) -> HeadAnchor | None:
    """The spec's own head body at its neck joint, or ``None`` without one.

    Anthropometric full-body specs carry a ``Head`` body on a three-axis neck
    at the cervicale; its frame origin is the neck point. The head length is
    the spec's de Leva head length when recorded.
    """
    if not isinstance(spec, Mapping) or "bodies" not in spec:
        raise ValueError("Spec must be a mapping with a 'bodies' list")
    for body in spec["bodies"]:
        name = str(body["name"])
        if name.lower().rsplit("/", 1)[-1] == "head":
            segments = (spec.get("anthropometry") or {}).get("segments") or {}
            length = float((segments.get("head") or {}).get("length_m", 0.0))
            return HeadAnchor(
                name, (0.0, 0.0, 0.0), length or DEFAULT_HEAD_LENGTH_M, "head_body"
            )
    return None


def _skull_point(direction: NDArray[np.float64], length: float) -> Array:
    """Point on the deformed skull for a unit direction (canonical frame)."""
    hx, hy, hz = (f * length for f in _SKULL_HALF)
    dx, dy, dz = direction
    low = max(0.0, -dz)  # 0 at/above the equator, 1 at the chin
    width = 1.0 - 0.30 * low**1.4  # jaw narrows toward the chin
    depth = 1.0 + (0.06 if dx < 0 else 0.0) * (1.0 - low)  # rounder back of skull
    chin = 0.10 * hx * low**2 * max(0.0, dx)  # chin carried forward
    return np.array(
        [
            hx * dx * depth + chin,
            hy * dy * width,
            _SKULL_CENTRE_Z * length + hz * dz,
        ]
    )


def _unit(theta: float, phi: float) -> NDArray[np.float64]:
    return np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ]
    )


def _skull(length: float) -> Mesh:
    thetas = np.linspace(0.0, math.pi, _RINGS + 2)[1:-1]
    phis = np.linspace(0.0, 2.0 * math.pi, _SIDES, endpoint=False)
    grid = np.array([[_skull_point(_unit(t, p), length) for p in phis] for t in thetas])
    return closed_grid_mesh(
        grid,
        _skull_point(np.array([0.0, 0.0, 1.0]), length),
        _skull_point(np.array([0.0, 0.0, -1.0]), length),
    )


def _face_point(length: float, azimuth: float, elevation: float, inset: float) -> Array:
    """Skull surface point at a facial direction, pulled in by ``inset``."""
    d = np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ]
    )
    p = _skull_point(d, length)
    centre = np.array([0.0, 0.0, _SKULL_CENTRE_Z * length])
    return centre + (p - centre) * (1.0 - inset)


_AX_Z = np.array([0.0, 0.0, 1.0])


def _blob(centre: Array, half: tuple[float, float, float], axis: Array) -> Mesh:
    """Ellipsoid with half sizes (along ``axis``, along x-ish, along y-ish)."""
    return ellipsoid_mesh(
        centre,
        np.array(half),
        axis,
        rings=10,
        sides=18,
        width_hint=np.array([1.0, 0.0, 0.0]),
    )


def _aligned(centre: Array, hx: float, hy: float, hz: float) -> Mesh:
    return _blob(centre, (hz, hx, hy), _AX_Z)


def _face_parts(length: float) -> list[HeadPart]:
    parts: list[HeadPart] = []
    for side, sign in (("l", 1.0), ("r", -1.0)):
        eye = _face_point(length, sign * 0.40, 0.14, 0.04)
        parts.append(
            HeadPart(
                f"eye_{side}",
                _aligned(eye, 0.050 * length, 0.062 * length, 0.050 * length),
                "eye_white",
            )
        )
        iris = eye + np.array([0.040 * length, 0.0, 0.0])
        parts.append(
            HeadPart(
                f"iris_{side}",
                _aligned(iris, 0.020 * length, 0.030 * length, 0.030 * length),
                "iris_dark",
            )
        )
        brow = _face_point(length, sign * 0.40, 0.36, 0.0)
        parts.append(
            HeadPart(
                f"brow_{side}",
                _aligned(brow, 0.026 * length, 0.085 * length, 0.016 * length),
                "brow_dark",
            )
        )
    nose_tip = _face_point(length, 0.0, -0.10, -0.12)
    tilt = np.array([0.30, 0.0, 1.0])
    parts.append(
        HeadPart(
            "nose",
            _blob(nose_tip, (0.115 * length, 0.075 * length, 0.045 * length), tilt),
            "skin",
        )
    )
    mouth = _face_point(length, 0.0, -0.46, -0.01)
    parts.append(
        HeadPart(
            "mouth",
            _aligned(mouth, 0.014 * length, 0.095 * length, 0.016 * length),
            "lip_pink",
        )
    )
    return parts


def _ears(length: float) -> list[HeadPart]:
    parts = []
    for side, sign in (("l", 1.0), ("r", -1.0)):
        centre = np.array(
            [-0.02 * length, sign * 0.285 * length, (_SKULL_CENTRE_Z - 0.03) * length]
        )
        parts.append(
            HeadPart(
                f"ear_{side}",
                _aligned(centre, 0.070 * length, 0.030 * length, 0.120 * length),
                "skin",
            )
        )
    return parts


def _neck(length: float) -> HeadPart:
    radius = 0.21 * length
    mesh = lofted_segment(
        np.array(
            [0.0, 0.0, 0.5 * radius - 0.004]
        ),  # spindle tip lands at the neck point
        np.array([0.0, 0.0, 0.40 * length]),
        radius,
        LoftOptions(
            aspect=1.08,
            coverage=(0.0, 1.0),
            rings=10,
            sides=28,
            width_hint=np.array([0.0, 1.0, 0.0]),
        ),
    )
    return HeadPart("neck", mesh, "skin")


def _dome(length: float, front: float, back: float, scale: float) -> Mesh:
    """Shell over the upper skull, cut at polar angle ``front`` ... ``back``."""
    phis = np.linspace(0.0, 2.0 * math.pi, _SIDES, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 16)[1:]
    centre = np.array([0.0, 0.0, _SKULL_CENTRE_Z * length])
    rows = []
    for u in thetas:
        row = []
        for phi in phis:
            cut = front + (back - front) * (1.0 - math.cos(phi)) / 2.0
            p = _skull_point(_unit(u * cut, phi), length)
            row.append(centre + (p - centre) * scale)
        rows.append(row)
    return closed_grid_mesh(
        np.array(rows),
        centre + (_skull_point(np.array([0.0, 0.0, 1.0]), length) - centre) * scale,
        centre * 1.0,
    )


def _headwear(length: float, kind: str) -> list[HeadPart]:
    if kind == "hair":
        return [HeadPart("hair", _dome(length, 0.95, 1.90, 1.05), "headwear")]
    brim_centre = np.array([0.43 * length, 0.0, (_SKULL_CENTRE_Z + 0.16) * length])
    visor = _aligned(brim_centre, 0.20 * length, 0.26 * length, 0.018 * length)
    return [
        HeadPart("cap", _dome(length, 1.0, 1.45, 1.06), "headwear"),
        HeadPart("visor", visor, "headwear"),
    ]


def build_head_parts(
    length_m: float, settings: HeadSettings | None = None
) -> list[HeadPart]:
    """Head, face, ears, neck and optional hair or cap, canonical frame.

    The frame has its origin at the neck point (cervicale), x forward, y left
    and z up; :func:`head_frame_in_body` places it in a body. Preconditions:
    positive finite head length. Postconditions: every mesh is closed with
    positive volume; parts mirror left/right about the mid-sagittal plane; the
    skull vertex is ``scale * length`` above the neck point.
    """
    if not math.isfinite(length_m) or length_m <= 0:
        raise ValueError("Head length must be a positive finite number")
    cfg = settings or HeadSettings()
    length = length_m * cfg.scale
    parts = [_neck(length), HeadPart("skull", _skull(length), "skin")]
    parts += _face_parts(length) + _ears(length)
    if cfg.headwear != "none":
        parts += _headwear(length, cfg.headwear)
    return parts


def head_frame_in_body(
    anchor: HeadAnchor, settings: HeadSettings | None = None
) -> Array:
    """4x4 pose of the canonical head frame in the anchor body's frame."""
    cfg = settings or HeadSettings()
    pose = np.eye(4)
    pose[:3, :3] = canonical_to_body(cfg.forward_axis, cfg.up_axis)
    pose[:3, 3] = anchor.origin_m
    return pose


def place_parts(parts: list[HeadPart], pose: Array) -> list[HeadPart]:
    """Parts with their vertices mapped through a 4x4 ``pose``."""
    p = np.asarray(pose, dtype=float)
    if p.shape != (4, 4) or not np.isfinite(p).all():
        raise ValueError("pose must be a finite 4x4 transform")
    return [
        HeadPart(
            part.name,
            Mesh(part.mesh.vertices @ p[:3, :3].T + p[:3, 3], part.mesh.faces.copy()),
            part.material,
        )
        for part in parts
    ]


def part_material_name(part: HeadPart, doc: AppearanceDocument) -> str:
    """Library material for a head part under the document's skin and headwear."""
    from src.shared.python.model_appearance import library

    if part.material == "skin":
        return doc.skin_tone
    if part.material == "headwear":
        name = library.headwear_material_name(doc)
        if name is None:
            raise ValueError("Headwear part requested for a bare head")
        return name
    return part.material
