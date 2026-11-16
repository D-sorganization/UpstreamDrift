"""Scene assembly and the deterministic camera rig of the hand renderer.

Pure NumPy, no GL: duck-typed backend results become :class:`ScenePart`
tuples, the anatomical ``(thumb, dorsum, shaft)`` basis is derived (or taken
from the caller), and the camera is auto-framed on skin vertices only.

All geometry is in the hand-grip frame of
:mod:`src.shared.python.model_appearance.grip_hand_mesh`: the grip axis is
``z`` (butt toward head) and the grip cylinder is ``x**2 + y**2 = R**2``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
)
from src.shared.python.model_appearance.club_shaft_mesh import tapered_tube
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.render_settings import (
    PART_KINDS,
    HandRenderSettings,
    Rgba,
)
from src.shared.python.model_appearance.library import MATERIALS

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ScenePart:
    """One mesh of the scene. ``kind`` is ``"skin"``, ``"grip"`` or ``"shaft"``.

    ``focus`` is ``None`` or a boolean mask, one entry per vertex, of the
    vertices the camera frames on. A skin part with a forearm stub marks the
    hand so the stub does not shrink the hand in the frame. ``name`` is an
    optional label that tells parts of one kind apart (the drawn grip uses
    :data:`GRIP_PART_NAMES`); it does not change how a part is rendered.
    """

    mesh: Any
    rgba: Rgba
    kind: str
    focus: Any = None
    name: str = ""


#: ``ScenePart.name`` of the rubber, the butt cap and the vent of a drawn grip.
GRIP_PART_NAMES: tuple[str, str, str] = ("grip_rubber", "grip_cap", "grip_vent")
#: ``ScenePart.name`` of the steel shaft of a drawn grip.
SHAFT_PART_NAME = "shaft"


@dataclass(frozen=True)
class GripDrawing:
    """Outline of a real golf grip: a tapered rubber sleeve with a butt cap.

    ``knots`` are ``(z_m, radius_m)`` pairs in the scene (club) frame with
    ``z`` strictly ascending; the radius is linear between them. The first
    knot is the edge of the butt cap and the last is the mouth (head end) of
    the grip. The cap flares from the first knot radius out to
    ``cap_radius_m`` and extends ``cap_height_m`` toward the butt (``-z``)
    beyond the first knot; ``cap_round_m`` is the radius of its rounded outer
    edge (zero is a sharp edge) and ``vent_radius_m`` the radius of the vent
    hole drawn in the centre of its flat end (zero is no vent).

    Contract (``ValueError`` otherwise): at least two knots, finite numbers,
    ascending ``z``, positive radii, ``cap_radius_m >= knots[0]`` radius,
    ``cap_height_m > 0``, ``0 <= cap_round_m <= cap_height_m`` and
    ``0 <= vent_radius_m < cap_radius_m - cap_round_m`` (the vent lies on the
    flat end face).
    """

    knots: tuple[tuple[float, float], ...]
    cap_radius_m: float
    cap_height_m: float
    cap_round_m: float
    vent_radius_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "knots", _checked_knots(self.knots))
        for name in ("cap_radius_m", "cap_height_m", "cap_round_m", "vent_radius_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            object.__setattr__(self, name, float(value))
        edge_radius = self.knots[0][1]
        if self.cap_radius_m < edge_radius:
            raise ValueError(
                f"cap_radius_m={self.cap_radius_m} must not be below the first "
                f"knot radius {edge_radius}"
            )
        if self.cap_height_m <= 0.0:
            raise ValueError(f"cap_height_m={self.cap_height_m} must be positive")
        if not 0.0 <= self.cap_round_m <= self.cap_height_m:
            raise ValueError(
                f"cap_round_m={self.cap_round_m} must lie in "
                f"[0, cap_height_m={self.cap_height_m}]"
            )
        flat = self.cap_radius_m - self.cap_round_m
        if not 0.0 <= self.vent_radius_m < flat:
            raise ValueError(
                f"vent_radius_m={self.vent_radius_m} must lie in [0, {flat}), "
                "inside the flat end of the cap (cap_radius_m - cap_round_m)"
            )

    @property
    def edge(self) -> tuple[float, float]:
        """``(z_m, radius_m)`` of the first knot, the edge of the butt cap."""
        return self.knots[0]

    @property
    def mouth(self) -> tuple[float, float]:
        """``(z_m, radius_m)`` of the last knot, the head end of the rubber."""
        return self.knots[-1]

    @property
    def cap_end_z_m(self) -> float:
        """``z`` of the flat end face of the butt cap."""
        return self.knots[0][0] - self.cap_height_m


def _checked_knots(knots: Any) -> tuple[tuple[float, float], ...]:
    """``knots`` as float pairs, validated for :class:`GripDrawing`."""
    if isinstance(knots, (str, bytes)) or not isinstance(knots, Sequence):
        raise ValueError("knots must be a sequence of (z_m, radius_m) pairs")
    if len(knots) < 2:
        raise ValueError(f"a grip drawing needs at least two knots, got {len(knots)}")
    out: list[tuple[float, float]] = []
    for knot in knots:
        if (
            isinstance(knot, (str, bytes))
            or not isinstance(knot, Sequence)
            or len(knot) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in knot)
        ):
            raise ValueError(f"knots must be (z_m, radius_m) pairs, got {knot!r}")
        z, radius = float(knot[0]), float(knot[1])
        if not (math.isfinite(z) and math.isfinite(radius)):
            raise ValueError(f"knots must be finite, got {knot!r}")
        if radius <= 0.0:
            raise ValueError(f"knot radii must be positive, got {knot!r}")
        if out and z <= out[-1][0]:
            raise ValueError(
                f"knot z must be strictly ascending, got {z} after {out[-1][0]}"
            )
        out.append((z, radius))
    return tuple(out)


@dataclass(frozen=True)
class AnatomicalBasis:
    """Orthonormal ``(thumb, dorsum, shaft)`` axes in the hand-grip frame.

    ``dorsum`` points out of the back of the hand, ``thumb`` toward the
    thumb (radial) side, ``shaft`` from the butt toward the head.
    """

    thumb: Array
    dorsum: Array
    shaft: Array

    def combine(self, coefficients: Sequence[float]) -> Array:
        """World vector of ``(thumb, dorsum, shaft)`` coefficients."""
        t, d, s = coefficients
        return t * self.thumb + d * self.dorsum + s * self.shaft


@dataclass(frozen=True)
class CameraRig:
    """A look-at camera. ``right``, ``up`` and ``back`` are the camera axes in
    the world (``back`` points from the target to the eye)."""

    eye: Array
    target: Array
    right: Array
    up: Array
    back: Array
    fov_deg: float
    distance: float

    def to_camera(self, points: Array) -> Array:
        """Points in camera coordinates relative to the eye (``z`` backward)."""
        rel = np.asarray(points, dtype=float) - self.eye
        return np.stack([rel @ self.right, rel @ self.up, rel @ self.back], axis=1)

    def project(self, points: Array) -> Array:
        """Normalised device ``(x, y)`` in ``[-1, 1]`` for a square frame."""
        cam = self.to_camera(points)
        scale = -cam[:, 2] * math.tan(math.radians(self.fov_deg) / 2.0)
        return cam[:, :2] / scale[:, None]

    def light_to_world(self, direction: Sequence[float]) -> Array:
        """A camera-space direction expressed in the world."""
        x, y, z = direction
        vec = x * self.right + y * self.up + z * self.back
        return vec / np.linalg.norm(vec)


@dataclass(frozen=True)
class RenderedView:
    """One rendered frame.

    ``image`` is ``uint8`` RGB of shape ``(frame_px, frame_px, 3)``;
    ``engine`` names what drew it (``osmesa``, ``egl`` or ``matplotlib``);
    ``fill_fraction`` is the measured share of the frame the skin spans on
    its longer axis. ``kind_mask`` (``uint8``, 0 background, then
    ``PART_KINDS`` order from 1) is ``None`` for the matplotlib fallback.
    """

    image: NDArray[np.uint8]
    engine: str
    fill_fraction: float
    view: str = ""
    camera: CameraRig | None = None
    kind_mask: NDArray[np.uint8] | None = None
    settings_hash: str = ""


def side_name(side: Any) -> str:
    """``"lead"`` or ``"trail"`` from a string or a :class:`Hand`."""
    value = side.value if isinstance(side, Hand) else side
    if value not in (Hand.LEAD.value, Hand.TRAIL.value):
        raise ValueError(f"side must be 'lead' or 'trail', got {side!r}")
    return str(value)


def as_mesh(mesh: Any) -> Mesh:
    """A validated :class:`Mesh` from any object with ``vertices`` and ``faces``."""
    if not (hasattr(mesh, "vertices") and hasattr(mesh, "faces")):
        raise TypeError(f"mesh needs .vertices and .faces, got {type(mesh).__name__}")
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(
            f"mesh vertices must be (N, 3) with N > 0, got {vertices.shape}"
        )
    if not np.isfinite(vertices).all():
        raise ValueError("mesh vertices must be finite")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"mesh faces must be (M, 3) with M > 0, got {faces.shape}")
    if not np.issubdtype(faces.dtype, np.integer):
        raise TypeError("mesh faces must be integers")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("mesh faces index outside the vertex array")
    return Mesh(vertices, faces.astype(np.int64))


def _rgba(value: Any, name: str) -> Rgba:
    arr: NDArray[np.float64] = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 3:
        arr = np.append(arr, 1.0)
    if arr.size != 4 or not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
        raise ValueError(f"{name} must be 3 or 4 numbers in [0, 1], got {value!r}")
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def validated_part(part: Any) -> ScenePart:
    """A :class:`ScenePart` with a checked mesh, colour and kind."""
    if not isinstance(part, ScenePart):
        raise TypeError(f"scene parts must be ScenePart, got {type(part).__name__}")
    if part.kind not in PART_KINDS:
        raise ValueError(f"part kind {part.kind!r} is not one of {PART_KINDS}")
    mesh = as_mesh(part.mesh)
    focus = checked_focus(part.focus, len(mesh.vertices))
    if not isinstance(part.name, str):
        raise TypeError(f"part name must be a string, got {part.name!r}")
    return ScenePart(mesh, _rgba(part.rgba, "part rgba"), part.kind, focus, part.name)


def checked_focus(focus: Any, vertex_count: int) -> NDArray[np.bool_] | None:
    """``focus`` as a boolean mask of ``vertex_count`` entries, or ``None``."""
    if focus is None:
        return None
    if isinstance(focus, str):
        raise TypeError("focus must be a boolean mask, not a string")
    mask = np.asarray(focus)
    if mask.dtype != np.bool_ or mask.shape != (vertex_count,):
        raise ValueError(
            f"focus must be a boolean mask of shape ({vertex_count},), "
            f"got {mask.dtype} {mask.shape}"
        )
    return mask


def result_meshes(result: Any) -> tuple[Mesh, ...]:
    """Skin meshes of a duck-typed backend result.

    Accepts an object with ``.mesh`` or a ``dict`` of meshes (one per bone).
    """
    if isinstance(result, Mapping):
        if not result:
            raise ValueError("the mesh dict is empty")
        return tuple(as_mesh(result[key]) for key in sorted(result, key=str))
    if hasattr(result, "mesh"):
        return (as_mesh(result.mesh),)
    if hasattr(result, "vertices") and hasattr(result, "faces"):
        return (as_mesh(result),)
    raise TypeError(
        "result must have .mesh, be a mesh, or be a dict of meshes; "
        f"got {type(result).__name__}"
    )


def result_rgba(result: Any, settings: HandRenderSettings) -> Rgba:
    """Skin colour of a backend result, or the settings fallback.

    Order: ``.skin_rgb``, ``.material_rgba``, ``.material`` (library name).
    """
    skin_rgb = getattr(result, "skin_rgb", None)
    if skin_rgb is not None:
        return _rgba(skin_rgb, "skin_rgb")
    material_rgba = getattr(result, "material_rgba", None)
    if material_rgba is not None:
        return _rgba(material_rgba, "material_rgba")
    material = getattr(result, "material", None)
    if material is not None:
        if not isinstance(material, str) or material not in MATERIALS:
            raise ValueError(f"material {material!r} is not a library material")
        return _rgba(MATERIALS[material].rgba, "material")
    return settings.skin_rgba


def checked_roll(shaft_roll_deg: Any, settings: HandRenderSettings) -> float:
    """A finite roll inside ``settings.roll_range_deg``."""
    if isinstance(shaft_roll_deg, bool) or not isinstance(
        shaft_roll_deg, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"shaft_roll_deg must be a number, got {shaft_roll_deg!r}")
    roll = float(shaft_roll_deg)
    low, high = settings.roll_range_deg
    if not math.isfinite(roll) or not low <= roll <= high:
        raise ValueError(
            f"shaft_roll_deg={shaft_roll_deg!r} is outside [{low}, {high}]"
        )
    return roll


def roll_mesh(mesh: Mesh, shaft_roll_deg: float) -> Mesh:
    """``mesh`` rolled about the grip axis by ``apply_shaft_roll``."""
    if shaft_roll_deg == 0.0:
        return mesh
    rotation = apply_shaft_roll(np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, shaft_roll_deg)
    return Mesh(mesh.vertices @ rotation.T, mesh.faces)


def build_club_parts(
    z_min: float,
    z_max: float,
    *,
    settings: HandRenderSettings,
    grip_radius_m: float | None = None,
    grip: GripDrawing | None = None,
) -> tuple[ScenePart, ...]:
    """Grip rubber and a shaft stub along ``+z`` for a hand spanning
    ``[z_min, z_max]``.

    Without ``grip`` the rubber is the cylinder ``x**2 + y**2 = R**2`` from
    ``z_min - grip_length_margin_m`` (butt side) to
    ``z_max + grip_length_margin_m``; the shaft continues
    ``shaft_extension_m`` toward the head. Returns ``(grip, shaft)``.

    With a :class:`GripDrawing` the whole grip is drawn from its butt cap to
    its mouth whatever ``[z_min, z_max]`` is, and ``grip_radius_m`` is not
    used: see :func:`drawn_club_parts` for the parts returned.

    Raises ``ValueError`` for non-finite or reversed ``z`` bounds or a radius
    that does not clear the shaft, ``TypeError`` for a ``grip`` that is not a
    :class:`GripDrawing`.
    """
    if grip is not None:
        return drawn_club_parts(z_min, z_max, grip, settings=settings)
    radius = settings.grip_radius_m if grip_radius_m is None else float(grip_radius_m)
    values = (float(z_min), float(z_max), radius)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("z_min, z_max and grip_radius_m must be finite")
    if z_min > z_max:
        raise ValueError(f"z_min={z_min} must not exceed z_max={z_max}")
    if radius <= settings.shaft_radius_m:
        raise ValueError("grip_radius_m must exceed settings.shaft_radius_m")
    butt = float(z_min) - settings.grip_length_margin_m
    rubber_end = float(z_max) + settings.grip_length_margin_m
    if rubber_end - butt <= settings.epsilon:
        raise ValueError("the grip needs a positive length; raise the margin")
    head = rubber_end + settings.shaft_extension_m
    sides = settings.club_sides
    ends = [np.array([0.0, 0.0, z]) for z in (butt, rubber_end, head)]
    rubber = tapered_tube(ends[0], ends[1], radius, radius, sides)
    shaft_r = settings.shaft_radius_m
    shaft = tapered_tube(ends[1], ends[2], shaft_r, shaft_r, sides)
    return (
        ScenePart(rubber, settings.grip_rgba, "grip"),
        ScenePart(shaft, settings.shaft_rgba, "shaft"),
    )


def _profile_strips(
    profile: Sequence[tuple[float, float]], *, crease_deg: float, epsilon: float
) -> list[list[tuple[float, float]]]:
    """``profile`` without repeated points, cut where it turns past the crease."""
    points: list[tuple[float, float]] = []
    for z, radius in profile:
        if (
            not points
            or math.hypot(z - points[-1][0], radius - points[-1][1]) > epsilon
        ):
            points.append((z, radius))
    strips = [[points[0]]]
    for index in range(1, len(points)):
        strips[-1].append(points[index])
        if index + 1 == len(points):
            break
        before = np.subtract(points[index], points[index - 1])
        after = np.subtract(points[index + 1], points[index])
        turn = math.atan2(
            abs(float(before[0] * after[1] - before[1] * after[0])),
            float(before @ after),
        )
        if math.degrees(turn) > crease_deg:
            strips.append([points[index]])
    return strips


def lathe_mesh(
    profile: Sequence[tuple[float, float]], *, settings: HandRenderSettings
) -> Mesh:
    """Closed solid of revolution about ``z`` through ``(z_m, radius_m)`` points.

    ``profile`` must outline a closed section: it starts and ends on the axis
    (radius zero) or returns to its first point. Each point becomes a ring of
    ``settings.club_sides`` vertices, or one pole vertex on the axis. The
    profile is cut where it turns by more than ``settings.normal_crease_deg``
    and the pieces do not share vertices, so smooth normals stay smooth along
    a taper and a hard rim stays sharp. Faces wind outward.

    Raises ``ValueError`` for fewer than three distinct points, a non-finite
    number, a negative radius, or a section that encloses no volume.
    """
    points = np.asarray(profile, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("profile must be finite (z_m, radius_m) pairs")
    if points[:, 1].min() < 0.0:
        raise ValueError("profile radii must not be negative")
    eps, sides = settings.epsilon, settings.club_sides
    strips = _profile_strips(
        [(float(z), float(r)) for z, r in points],
        crease_deg=settings.normal_crease_deg,
        epsilon=eps,
    )
    if sum(len(strip) for strip in strips) < 3:
        raise ValueError("profile needs at least three distinct points")
    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    circle = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    step = np.arange(sides)
    after = (step + 1) % sides
    vertices: list[Array] = []
    faces: list[NDArray[np.int64]] = []
    count = 0
    for strip in strips:
        rows: list[tuple[int, bool]] = []
        for z, radius in strip:
            pole = radius <= eps
            ring = np.zeros((1, 2)) if pole else radius * circle
            vertices.append(np.column_stack([ring, np.full(len(ring), z)]))
            rows.append((count, pole))
            count += len(ring)
        for (a, a_pole), (b, b_pole) in zip(rows, rows[1:], strict=False):
            if a_pole and b_pole:
                raise ValueError("profile runs along the axis")
            if a_pole:
                faces.append(np.stack([np.full(sides, a), b + step, b + after], axis=1))
            elif b_pole:
                faces.append(np.stack([a + step, np.full(sides, b), a + after], axis=1))
            else:
                faces.append(np.stack([a + step, b + step, b + after], axis=1))
                faces.append(np.stack([a + step, b + after, a + after], axis=1))
    mesh = Mesh(np.vstack(vertices), np.vstack(faces).astype(np.int64))
    volume = mesh.volume()
    if abs(volume) <= eps**2:
        raise ValueError("profile encloses no volume")
    if volume < 0.0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, ::-1].copy())
    return mesh


def _rubber_profile(
    grip: GripDrawing, settings: HandRenderSettings
) -> list[tuple[float, float]]:
    """Rubber section: butt closure, the knots (subdivided), mouth closure."""
    profile = [(grip.edge[0], 0.0), grip.edge]
    for (z_a, r_a), (z_b, r_b) in zip(grip.knots, grip.knots[1:], strict=False):
        rings = max(1, math.ceil((z_b - z_a) / settings.grip_ring_spacing_m))
        for index in range(1, rings):
            t = index / rings
            profile.append((z_a + t * (z_b - z_a), r_a + t * (r_b - r_a)))
        profile.append((z_b, r_b))
    profile.append((grip.mouth[0], 0.0))
    return profile


def _cap_profile(
    grip: GripDrawing, settings: HandRenderSettings
) -> list[tuple[float, float]]:
    """Cap section: flare, rounded edge, flat end and (with a vent) its bore."""
    z_edge, r_edge = grip.edge
    end, vent = grip.cap_end_z_m, grip.vent_radius_m
    wide, rounding = grip.cap_radius_m, grip.cap_round_m
    profile = [(z_edge, vent), (z_edge, r_edge), (end + rounding, wide)]
    segments = settings.grip_cap_round_segments
    for index in range(1, segments + 1):
        angle = 0.5 * math.pi * index / segments
        profile.append(
            (
                end + rounding * (1.0 - math.sin(angle)),
                wide - rounding * (1.0 - math.cos(angle)),
            )
        )
    profile[-1] = (end, wide - rounding)
    profile.append((end, vent))
    if vent > 0.0:
        profile.append((z_edge, vent))
    return profile


def drawn_club_parts(
    z_min: float, z_max: float, grip: GripDrawing, *, settings: HandRenderSettings
) -> tuple[ScenePart, ...]:
    """A real golf grip and its steel shaft for a hand spanning ``[z_min, z_max]``.

    Returns the parts named :data:`GRIP_PART_NAMES` then
    :data:`SHAFT_PART_NAME`: the rubber (a surface of revolution through
    ``grip.knots`` with a ring at least every ``grip_ring_spacing_m``), the
    butt cap (flare, rounded edge, flat end), the vent plug that fills the
    hole in the cap end (only when ``grip.vent_radius_m > 0``), and the
    shaft. The first three are of kind ``"grip"``. The shaft starts at the
    mouth, ``grip_mouth_step_m`` thinner than the rubber there, and runs to
    the farther of ``z_max + grip_length_margin_m + shaft_extension_m`` and
    ``shaft_extension_m`` beyond the mouth, so it never shows at the butt.

    Raises ``TypeError`` when ``grip`` is not a :class:`GripDrawing`,
    ``ValueError`` for non-finite or reversed ``z`` bounds or a mouth too
    thin for ``grip_mouth_step_m``.
    """
    if not isinstance(grip, GripDrawing):
        raise TypeError(f"grip must be a GripDrawing or None, got {grip!r}")
    if not (math.isfinite(z_min) and math.isfinite(z_max)):
        raise ValueError("z_min and z_max must be finite")
    if z_min > z_max:
        raise ValueError(f"z_min={z_min} must not exceed z_max={z_max}")
    mouth_z, mouth_radius = grip.mouth
    shaft_radius = mouth_radius - settings.grip_mouth_step_m
    if shaft_radius <= 0.0:
        raise ValueError(
            f"grip_mouth_step_m={settings.grip_mouth_step_m} leaves no shaft "
            f"inside a mouth of radius {mouth_radius}"
        )
    rubber_name, cap_name, vent_name = GRIP_PART_NAMES
    rubber = lathe_mesh(_rubber_profile(grip, settings), settings=settings)
    cap = lathe_mesh(_cap_profile(grip, settings), settings=settings)
    parts = [
        ScenePart(rubber, settings.grip_rgba, "grip", name=rubber_name),
        ScenePart(cap, settings.grip_cap_rgba, "grip", name=cap_name),
    ]
    if grip.vent_radius_m > 0.0:
        end, vent = grip.cap_end_z_m, grip.vent_radius_m
        plug = lathe_mesh(
            [(end, 0.0), (end, vent), (grip.edge[0], vent), (grip.edge[0], 0.0)],
            settings=settings,
        )
        parts.append(ScenePart(plug, settings.grip_vent_rgba, "grip", name=vent_name))
    reach = settings.shaft_extension_m
    head = max(float(z_max) + settings.grip_length_margin_m, mouth_z) + reach
    shaft = tapered_tube(
        np.array([0.0, 0.0, mouth_z]),
        np.array([0.0, 0.0, head]),
        shaft_radius,
        shaft_radius,
        settings.club_sides,
    )
    parts.append(ScenePart(shaft, settings.shaft_rgba, "shaft", name=SHAFT_PART_NAME))
    ensure(
        float(shaft.vertices[:, 2].min()) >= mouth_z - settings.epsilon,
        "the shaft is drawn only beyond the mouth of the grip",
    )
    return tuple(parts)


def assemble_scene(
    result: Any,
    *,
    shaft_roll_deg: float = 0.0,
    settings: HandRenderSettings,
    grip_radius_m: float | None = None,
) -> tuple[ScenePart, ...]:
    """Skin, grip and shaft parts for one backend result.

    The roll turns the skin only (the club is a body of revolution). The grip
    radius is ``grip_radius_m``, else ``result.grip_radius_m``, else settings.
    A single-mesh result may declare ``frame_focus``, the vertex mask the
    camera frames on (see :class:`ScenePart`).
    """
    roll = checked_roll(shaft_roll_deg, settings)
    meshes = result_meshes(result)
    rgba = result_rgba(result, settings)
    radius = grip_radius_m
    if radius is None:
        radius = getattr(result, "grip_radius_m", None)
    focus = getattr(result, "frame_focus", None) if len(meshes) == 1 else None
    skin = tuple(ScenePart(roll_mesh(m, roll), rgba, "skin", focus) for m in meshes)
    z = np.concatenate([m.vertices[:, 2] for m in meshes])
    club = build_club_parts(
        float(z.min()), float(z.max()), settings=settings, grip_radius_m=radius
    )
    parts = (*skin, *club)
    ensure(
        {p.kind for p in parts} == set(PART_KINDS),
        "an assembled scene holds skin, grip and shaft parts",
    )
    return parts


def skin_vertices_of(parts: Sequence[ScenePart]) -> Array:
    """All skin vertices of a scene, stacked. Raises when there is no skin."""
    skins = [
        np.asarray(p.mesh.vertices, dtype=float) for p in parts if p.kind == "skin"
    ]
    if not skins:
        raise ValueError("the scene has no skin part to frame the camera on")
    return np.vstack(skins)


def focus_vertices_of(parts: Sequence[ScenePart]) -> Array:
    """The skin vertices the camera frames on, stacked.

    A skin part with a ``focus`` mask contributes only its marked vertices; a
    part without one contributes all of them. Raises ``ValueError`` when the
    scene has no skin or every skin vertex is masked out.
    """
    chosen = []
    for part in parts:
        if part.kind != "skin":
            continue
        vertices = np.asarray(part.mesh.vertices, dtype=float)
        mask = checked_focus(part.focus, len(vertices))
        chosen.append(vertices if mask is None else vertices[mask])
    if not chosen:
        raise ValueError("the scene has no skin part to frame the camera on")
    stacked = np.vstack(chosen)
    if len(stacked) == 0:
        raise ValueError("the focus masks leave no skin vertex to frame on")
    return stacked


def _unit(vector: Any, name: str, epsilon: float) -> Array:
    arr = np.asarray(vector, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 3-vector, got {vector!r}")
    norm = float(np.linalg.norm(arr))
    if norm <= epsilon:
        raise ValueError(f"{name} is degenerate (zero, or parallel to another axis)")
    return arr / norm


def orthonormal_basis(
    thumb: Any, dorsum: Any, shaft: Any = None, *, epsilon: float
) -> AnatomicalBasis:
    """Gram-Schmidt in the order shaft, dorsum, thumb.

    The shaft is kept, the dorsum loses its shaft component, and the thumb
    loses both. A thumb direction that lies along the shaft is degenerate:
    pass the radial (thumb-side) direction of the hand, not the thumb bone.
    """
    s = _unit(APPEARANCE_GRIP_SHAFT_AXIS if shaft is None else shaft, "shaft", epsilon)
    d_raw = _unit(dorsum, "dorsum", epsilon)
    d = _unit(d_raw - (d_raw @ s) * s, "dorsum", epsilon)
    t_raw = _unit(thumb, "thumb", epsilon)
    t = _unit(t_raw - (t_raw @ s) * s - (t_raw @ d) * d, "thumb", epsilon)
    return AnatomicalBasis(thumb=t, dorsum=d, shaft=s)


def anatomical_basis(
    skin_vertices: Any, *, side: Any, settings: HandRenderSettings
) -> AnatomicalBasis:
    """Heuristic basis from skin vertices in the hand-grip frame.

    Shaft is ``+z``. Dorsum is the ``xy`` direction of the centroid of the
    outer skin (radius above ``basis.outer_radius_quantile``): the bulk of a
    gripping hand sits on the back-of-hand side of the shaft. Thumb is the
    ``xy`` direction of the highest skin (``z`` above
    ``basis.thumb_height_quantile``) made perpendicular to the dorsum; when
    that is degenerate it falls back to ``cross(dorsum, shaft)`` signed by
    ``side``. A backend that knows its own axes should pass them instead.
    """
    sign = 1.0 if side_name(side) == Hand.LEAD.value else -1.0
    vertices = np.asarray(skin_vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(
            f"skin_vertices must be (N, 3) with N > 0, got {vertices.shape}"
        )
    if not np.isfinite(vertices).all():
        raise ValueError("skin_vertices must be finite")
    eps = settings.epsilon
    shaft = np.asarray(APPEARANCE_GRIP_SHAFT_AXIS, dtype=float)
    radius = np.hypot(vertices[:, 0], vertices[:, 1])
    outer = radius >= np.quantile(radius, settings.basis_outer_radius_quantile)
    dorsum_xy = np.append(vertices[outer, :2].mean(axis=0), 0.0)
    if np.linalg.norm(dorsum_xy) <= eps:
        raise ValueError("cannot derive the dorsum: the skin is centred on the shaft")
    top = vertices[:, 2] >= np.quantile(
        vertices[:, 2], settings.basis_thumb_height_quantile
    )
    thumb_xy = np.append(vertices[top, :2].mean(axis=0), 0.0)
    fallback = sign * np.cross(dorsum_xy, shaft)
    try:
        basis = orthonormal_basis(thumb_xy, dorsum_xy, shaft, epsilon=math.sqrt(eps))
    except ValueError:
        basis = orthonormal_basis(fallback, dorsum_xy, shaft, epsilon=eps)
    return basis


def result_basis(
    result: Any, *, settings: HandRenderSettings
) -> AnatomicalBasis | None:
    """A basis the result itself declares, or ``None``.

    Order: ``result.render_basis`` (an :class:`AnatomicalBasis` or
    ``(thumb, dorsum[, shaft])``), then named ``result.components`` whose
    ``finger`` tags include ``"dorsum"`` and ``"thumb"`` (the capsule hand):
    the dorsum and thumb axes are the ``xy`` directions of those shells.
    """
    declared = getattr(result, "render_basis", None)
    if declared is not None:
        return coerce_basis(
            declared, np.zeros((1, 3)), side=Hand.LEAD, settings=settings
        )
    components = getattr(result, "components", None)
    if not components:
        return None
    centroids: dict[str, list[Array]] = {"dorsum": [], "thumb": []}
    for component in components:
        tag = getattr(component, "finger", None)
        if tag in centroids and hasattr(component, "mesh"):
            centroids[tag].append(as_mesh(component.mesh).vertices)
    if not (centroids["dorsum"] and centroids["thumb"]):
        return None
    dorsum = np.append(np.vstack(centroids["dorsum"])[:, :2].mean(axis=0), 0.0)
    thumb = np.append(np.vstack(centroids["thumb"])[:, :2].mean(axis=0), 0.0)
    try:
        return orthonormal_basis(thumb, dorsum, epsilon=math.sqrt(settings.epsilon))
    except ValueError:
        return None


def coerce_basis(
    basis: Any,
    skin_vertices: Array,
    *,
    side: Any,
    settings: HandRenderSettings,
) -> AnatomicalBasis:
    """The caller's basis, orthonormalised, or the vertex heuristic for ``None``.

    ``basis`` may be an :class:`AnatomicalBasis` or a sequence
    ``(thumb, dorsum)`` / ``(thumb, dorsum, shaft)`` of 3-vectors.
    """
    if basis is None:
        return anatomical_basis(skin_vertices, side=side, settings=settings)
    if isinstance(basis, AnatomicalBasis):
        axes: tuple[Any, ...] = (basis.thumb, basis.dorsum, basis.shaft)
    elif isinstance(basis, Sequence) and len(basis) in (2, 3):
        axes = tuple(basis)
    else:
        raise TypeError(
            "basis must be None, an AnatomicalBasis, or (thumb, dorsum[, shaft])"
        )
    return orthonormal_basis(*axes, epsilon=settings.epsilon)


def camera_rig(
    skin_vertices: Any,
    view: str,
    basis: AnatomicalBasis,
    settings: HandRenderSettings,
) -> CameraRig:
    """Look-at camera for ``view``, auto-framed on the skin vertices.

    The eye sits along the view's ``eye`` coefficients in the anatomical
    basis. The distance and target are iterated so the projected skin bounds
    are centred and their longer axis spans ``settings.fill_fraction`` of the
    frame. Deterministic: the same inputs give the same rig.
    """
    spec = settings.view(view)
    vertices = np.asarray(skin_vertices, dtype=float)
    require(
        vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) > 0,
        "skin_vertices is (N, 3) with N > 0",
    )
    eps = settings.epsilon
    back = _unit(basis.combine(spec.eye), f"view {view!r} eye", eps)
    up_hint = basis.combine(spec.up)
    right = _unit(np.cross(up_hint, back), f"view {view!r} up", eps)
    up: Array = np.asarray(np.cross(back, right), dtype=np.float64)
    axes = np.stack([right, up, back], axis=1)
    cam = vertices @ axes
    lo, hi = cam.min(axis=0), cam.max(axis=0)
    centre = (lo + hi) / 2.0
    span = settings.fill_fraction * math.tan(math.radians(settings.fov_deg) / 2.0)

    def fit(centre_now: Array) -> float:
        lateral = np.abs(cam[:, :2] - centre_now[:2]).max(axis=1)
        return float(np.max(lateral / span + cam[:, 2] - centre_now[2]))

    for _ in range(settings.frame_iterations):
        distance = fit(centre)
        depth = distance - (cam[:, 2] - centre[2])
        ndc = (cam[:, :2] - centre[:2]) / (depth[:, None] * span)
        shift = (ndc.min(axis=0) + ndc.max(axis=0)) / 2.0
        centre = centre + np.append(shift * span * distance, 0.0)
    distance = fit(centre)
    target = axes @ centre
    rig = CameraRig(
        eye=target + distance * back,
        target=target,
        right=right,
        up=up,
        back=back,
        fov_deg=settings.fov_deg,
        distance=distance,
    )
    ensure(bool(distance > 0.0 and np.isfinite(rig.eye).all()), "the eye is finite")
    return rig


def projected_fill(rig: CameraRig, skin_vertices: Any) -> float:
    """Share of a square frame the projected skin spans on its longer axis."""
    ndc = rig.project(np.asarray(skin_vertices, dtype=float))
    return float(np.max(ndc.max(axis=0) - ndc.min(axis=0)) / 2.0)


def vertex_normals(mesh: Mesh, *, weld_tolerance_m: float) -> Array:
    """Smooth unit vertex normals, area-weighted, welded by position.

    Vertices closer than ``weld_tolerance_m`` share one normal, so a mesh
    with duplicated seam vertices still shades without a crease.
    """
    vertices, faces = mesh.vertices, mesh.faces
    tri = vertices[faces]
    face_normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    keys = np.round(vertices / weld_tolerance_m).astype(np.int64)
    _, welded = np.unique(keys, axis=0, return_inverse=True)
    welded = welded.reshape(-1)
    summed = np.zeros((int(welded.max()) + 1, 3))
    for corner in range(3):
        np.add.at(summed, welded[faces[:, corner]], face_normal)
    normals = summed[welded]
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = np.array(APPEARANCE_GRIP_SHAFT_AXIS, dtype=float)
    normals = np.where(
        length > 0.0, normals / np.where(length > 0.0, length, 1.0), fallback
    )
    ensure(bool(np.isfinite(normals).all()), "vertex normals are finite")
    return normals


def corner_normals(mesh: Mesh, *, weld_tolerance_m: float, crease_deg: float) -> Array:
    """Smooth unit normals per face corner, shape ``(M, 3, 3)``.

    Like :func:`vertex_normals`, but a corner only averages the faces around
    its (welded) vertex that lie within ``crease_deg`` of its own face. Skin
    stays smooth; a hard edge such as a block rim or a wrist cut stays sharp
    instead of smearing a gradient across the flat sides.
    """
    vertices, faces = mesh.vertices, mesh.faces
    tri = vertices[faces]
    weighted = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(weighted, axis=1, keepdims=True)
    unit = weighted / np.where(area > 0.0, area, 1.0)
    keys = np.round(vertices / weld_tolerance_m).astype(np.int64)
    _, welded = np.unique(keys, axis=0, return_inverse=True)
    corner_vertex = welded.reshape(-1)[faces].reshape(-1)
    corner_face = np.repeat(np.arange(len(faces)), 3)
    order = np.argsort(corner_vertex, kind="stable")
    _, first, count = np.unique(
        corner_vertex[order], return_index=True, return_counts=True
    )
    group_start = np.repeat(first, count)
    group_size = np.repeat(count, count)
    source = np.repeat(np.arange(len(order)), group_size)
    block_start = np.cumsum(group_size) - group_size
    partner = np.repeat(group_start, group_size) + (
        np.arange(len(source)) - np.repeat(block_start, group_size)
    )
    face_a = corner_face[order][source]
    face_b = corner_face[order][partner]
    cosine = np.einsum("ij,ij->i", unit[face_a], unit[face_b])
    keep = cosine >= math.cos(math.radians(crease_deg))
    summed = np.zeros((len(order), 3))
    np.add.at(summed, source[keep], weighted[face_b[keep]])
    normals: NDArray[np.float64] = np.empty_like(summed)
    normals[order] = summed
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    own = unit[corner_face]
    normals = np.where(length > 0.0, normals / np.where(length > 0.0, length, 1.0), own)
    ensure(bool(np.isfinite(normals).all()), "corner normals are finite")
    return normals.reshape(len(faces), 3, 3)
