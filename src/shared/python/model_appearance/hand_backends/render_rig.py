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
    """One mesh of the scene. ``kind`` is ``"skin"``, ``"grip"`` or ``"shaft"``."""

    mesh: Any
    rgba: Rgba
    kind: str


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
    arr = np.asarray(value, dtype=float).reshape(-1)
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
    return ScenePart(as_mesh(part.mesh), _rgba(part.rgba, "part rgba"), part.kind)


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
) -> tuple[ScenePart, ScenePart]:
    """Grip rubber and a shaft stub along ``+z`` for a hand spanning
    ``[z_min, z_max]``.

    The rubber is the cylinder ``x**2 + y**2 = R**2`` from
    ``z_min - grip_length_margin_m`` (butt side) to
    ``z_max + grip_length_margin_m``; the shaft continues
    ``shaft_extension_m`` toward the head. Returns ``(grip, shaft)``.
    """
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
    grip = tapered_tube((0, 0, butt), (0, 0, rubber_end), radius, radius, sides)
    shaft_r = settings.shaft_radius_m
    shaft = tapered_tube((0, 0, rubber_end), (0, 0, head), shaft_r, shaft_r, sides)
    return (
        ScenePart(grip, settings.grip_rgba, "grip"),
        ScenePart(shaft, settings.shaft_rgba, "shaft"),
    )


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
    """
    roll = checked_roll(shaft_roll_deg, settings)
    meshes = result_meshes(result)
    rgba = result_rgba(result, settings)
    radius = grip_radius_m
    if radius is None:
        radius = getattr(result, "grip_radius_m", None)
    skin = tuple(ScenePart(roll_mesh(m, roll), rgba, "skin") for m in meshes)
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
    up = np.cross(back, right)
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
    ensure(distance > 0.0 and np.isfinite(rig.eye).all(), "the eye is finite")
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
