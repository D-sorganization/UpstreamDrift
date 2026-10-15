"""Engine-neutral body meshes: the shared appearance and anthropometry spec.

R8 (#12685, epic #12678). One resolver turns a body/joint spec plus an
appearance document into the skin, garment, blend and shoe meshes of every
body, so each engine emitter only *translates* them (MuJoCo today; Drake,
Pinocchio, OpenSim/MyoSuite and Simscape in R9a/b/c). Emitters receive a
:class:`ResolvedAppearance` and never read the classification, sizing or
coverage tables, nor the spec internals (Law of Demeter).

Frame contract (DbC)
    Every ``BodyMesh.mesh_in_body_frame`` vertex is expressed in the **spec
    body frame** of ``BodyMesh.body`` (:data:`SPEC_BODY_FRAME`): the frame
    the spec's joints, ``visual_skeleton`` capsules and ``visual_hints``
    shapes are written in. It is *not* an engine body frame. An engine whose
    body frame differs must transform explicitly: the MuJoCo exporter's body
    is the joint follower frame, so its emitter maps every vertex through
    ``inv(child_to_follower)``. The v1 chest (``COMRod``) has a non-identity
    follower rotation; drawing these meshes in it untransformed misorients
    the chest.

Sizing rule
    Limbs, hands, feet and the head use ``library.PART_RADIUS_M``. The trunk
    (torso, pelvis) is ONE lofted solid per body along its longest capsule
    (joint origin to the farthest child joint), sized from the spec's own
    ``visual_hints`` ellipsoids: radius = the largest mediolateral (ML)
    semi-axis, aspect = the largest anteroposterior (AP) semi-axis over it,
    with ML/AP from the spec landmarks (``cross_section``). A trunk body
    without hint shapes uses the documented library default
    (``source="library_default"``). See
    ``docs/development/full_body_models/appearance_reference/shared_appearance_spec.tex``.

Head and club meshes stay in their own shared modules (``head``,
``club_assembly``/``grip_hands``); a head mesh reference slot is reserved for
R7 (#12687). Visual only: nothing here touches mass, inertia or joints.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.model_appearance import geometry, library
from src.shared.python.model_appearance.club_assembly import (
    PART_MATERIALS,
    ClubAssembly,
)
from src.shared.python.model_appearance.cross_section import (
    TRUNK_PARTS,
    BodySection,
    body_sections,
    width_hint_for,
)
from src.shared.python.model_appearance.grip_hands import assembly_meshes_with_hands
from src.shared.python.model_appearance.schema import AppearanceDocument, Material
from src.shared.python.motion_matching.visual_skeleton import (
    Capsule,
    Shape,
    derive_visual_skeleton,
)

__all__ = [
    "SPEC_BODY_FRAME",
    "BodyMesh",
    "MeshReference",
    "ResolvedAppearance",
    "SectionSize",
    "body_meshes",
    "club_meshes",
    "resolve_appearance",
    "section_size",
]

SPEC_BODY_FRAME = "spec_body"
SIZING_SOURCES = ("visual_hints", "library_default")
# A short link's loft radius is clamped to this fraction of its length so it
# does not balloon (library-default sizing only; hint sizes are anthropometric).
_MAX_RADIUS_OVER_LENGTH = 0.6
_BLEND_RINGS, _BLEND_SIDES = 14, 24


@dataclass(frozen=True)
class SectionSize:
    """Trunk cross-section: ML semi-axis, AP/ML aspect and where it came from."""

    radius_m: float
    aspect: float
    source: str

    def __post_init__(self) -> None:
        if not (math.isfinite(self.radius_m) and self.radius_m > 0):
            raise ValueError(f"section radius must be positive, got {self.radius_m}")
        if not (math.isfinite(self.aspect) and 0 < self.aspect <= 1):
            raise ValueError(
                f"section aspect AP/ML must be in (0, 1] (a trunk deeper than "
                f"wide is a spec error), got {self.aspect}"
            )
        if self.source not in SIZING_SOURCES:
            raise ValueError(f"sizing source must be one of {SIZING_SOURCES}")


@dataclass(frozen=True)
class BodyMesh:
    """One visual mesh of one body, in that body's SPEC frame."""

    body: str
    label: str  # skin<i>, blend<i>, garment<i> or a club part name
    mesh_in_body_frame: geometry.Mesh
    material_name: str

    def __post_init__(self) -> None:
        for name in ("body", "label", "material_name"):
            if not getattr(self, name):
                raise ValueError(f"BodyMesh.{name} must be a non-empty string")
        verts = np.asarray(self.mesh_in_body_frame.vertices)
        if verts.ndim != 2 or verts.shape[1] != 3 or not np.isfinite(verts).all():
            raise ValueError(f"BodyMesh {self.body}/{self.label}: bad vertices")


@dataclass(frozen=True)
class MeshReference:
    """A licensed external mesh (slot for the R7 head, #12687)."""

    path: str
    licence: str
    provenance: str

    def __post_init__(self) -> None:
        for name in ("path", "licence", "provenance"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"MeshReference.{name} must be a non-empty string")


@dataclass(frozen=True)
class ResolvedAppearance:
    """Everything an engine emitter needs, already resolved (LoD boundary).

    ``meshes`` are in :data:`SPEC_BODY_FRAME`; ``materials`` holds every
    material a mesh names; ``head_mesh`` is the R7 head slot (``None`` keeps
    the parametric ``model_appearance.head``).
    """

    meshes: tuple[BodyMesh, ...]
    materials: Mapping[str, Material]
    head_mesh: MeshReference | None = None
    frame: str = field(default=SPEC_BODY_FRAME)

    def meshes_for(self, body: str) -> tuple[BodyMesh, ...]:
        return tuple(m for m in self.meshes if m.body == body)

    def rgba(self, material_name: str) -> tuple[float, float, float, float]:
        if material_name not in self.materials:
            raise ValueError(f"Material {material_name!r} is not in the resolution")
        return self.materials[material_name].rgba


def _semi_axis(shape: Shape, direction: np.ndarray) -> float:
    """Half extent of an axis-aligned hint shape along a unit ``direction``."""
    half = np.asarray(shape.half_size_m, dtype=float)
    d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    if shape.kind == "box":
        return float(np.abs(d) @ half)
    return float(1.0 / np.sqrt(np.sum((d / half) ** 2)))


def _section_size(
    part: str, shapes: Sequence[Shape], section: BodySection | None
) -> SectionSize:
    if not shapes:
        return SectionSize(
            library.PART_RADIUS_M[part], library.PART_ASPECT[part], "library_default"
        )
    if section is None:
        raise ValueError(f"a hinted {part} body needs landmark axes (cross_section)")
    ml = max(_semi_axis(s, section.ml_axis) for s in shapes)
    ap = max(_semi_axis(s, section.depth_axis) for s in shapes)
    return SectionSize(ml, ap / ml, "visual_hints")


def section_size(spec: Mapping[str, Any], body: str) -> SectionSize:
    """Trunk cross-section of ``body`` under the sizing rule (module docstring).

    Precondition: ``body`` classifies as a trunk part. Postcondition: the ML
    semi-axis and AP/ML aspect of the hint ellipsoids, or the library default
    when the body has no hint shapes.
    """
    part = library.classify_body(body)
    if part not in TRUNK_PARTS:
        raise ValueError(f"{body!r} is a {part!r}, not a trunk body")
    shapes = [s for s in derive_visual_skeleton(spec).shapes if s.body == body]
    sections = body_sections(spec) if shapes else {}
    return _section_size(part, shapes, sections.get(body))


def club_meshes(
    club: ClubAssembly, finish: str
) -> list[tuple[str, geometry.Mesh, str]]:
    """Shaft, grip, mesh head (GCV-11) and both grip hands (OSV-2), club frame."""
    return [
        (label, mesh, PART_MATERIALS[label] or finish)
        for label, mesh in assembly_meshes_with_hands(club).items()
    ]


@dataclass(frozen=True)
class _Context:
    """Per-resolution inputs shared by every body."""

    doc: AppearanceDocument
    sections: Mapping[str, BodySection]
    shapes: Mapping[str, Sequence[Shape]]
    club: ClubAssembly | None


@dataclass(frozen=True)
class _Segment:
    start: np.ndarray
    end: np.ndarray
    radius: float
    aspect: float


def _limb_segments(part: str, caps: Sequence[Capsule], scale: float) -> list[_Segment]:
    out = []
    for cap in caps:
        radius = scale * (
            cap.radius_m
            if part in ("club", "other")
            else library.PART_RADIUS_M.get(part, cap.radius_m)
        )
        radius = min(radius, _MAX_RADIUS_OVER_LENGTH * cap.length_m())
        aspect = library.PART_ASPECT.get(part, 1.0)
        out.append(
            _Segment(np.asarray(cap.start_m), np.asarray(cap.end_m), radius, aspect)
        )
    return out


def _trunk_segment(
    ctx: _Context, body: str, part: str, caps: Sequence[Capsule], scale: float
) -> _Segment:
    """One trunk solid along the body's longest capsule (no overlapping lofts)."""
    spine = max(caps, key=lambda c: c.length_m())  # first wins on a tie
    size = _section_size(part, ctx.shapes.get(body, ()), ctx.sections.get(body))
    radius = scale * size.radius_m
    if size.source == "library_default":
        radius = min(radius, _MAX_RADIUS_OVER_LENGTH * spine.length_m())
    return _Segment(
        np.asarray(spine.start_m), np.asarray(spine.end_m), radius, size.aspect
    )


def _loft(ctx: _Context, body: str, seg: _Segment, **extra: Any) -> geometry.Mesh:
    return geometry.lofted_segment(
        seg.start,
        seg.end,
        seg.radius,
        geometry.LoftOptions(
            aspect=seg.aspect,
            rings=library.LOFT_RINGS,
            sides=library.LOFT_SIDES,
            width_hint=width_hint_for(ctx.sections, body, seg.start, seg.end),
            **extra,
        ),
    )


def _segment_meshes(
    ctx: _Context, body: str, part: str, segments: Sequence[_Segment]
) -> list[BodyMesh]:
    base = library.material_name_for(ctx.doc, body)
    garment = library.garment_name_for(ctx.doc, body)
    out = []
    for i, seg in enumerate(segments):
        out.append(BodyMesh(body, f"skin{i}", _loft(ctx, body, seg), base))
        if part in library.BLEND_PARTS:  # ball joint hides the segment seam
            ball = geometry.ellipsoid_mesh(
                seg.start,
                np.full(3, seg.radius * library.blend_scale(garment is not None)),
                seg.end - seg.start,
                rings=_BLEND_RINGS,
                sides=_BLEND_SIDES,
            )
            out.append(BodyMesh(body, f"blend{i}", ball, garment or base))
        if garment is not None:
            band = library.garment_coverage(ctx.doc.clothing, part)
            mesh = _loft(
                ctx, body, seg, coverage=band, thickness=library.GARMENT_THICKNESS
            )
            out.append(BodyMesh(body, f"garment{i}", mesh, garment))
    return out


def _meshes_for_body(
    ctx: _Context, body: str, caps: Sequence[Capsule]
) -> list[BodyMesh]:
    rule = library.rule_for_body(ctx.doc, body)
    if rule is not None and rule.mesh == "none":
        return []
    part = library.classify_body(body)
    if part == "head" and ctx.doc.head.enabled:
        return []  # drawn by the shared head: skull, face, ears and neck
    if part == "club" and ctx.club is not None:
        finish = library.material_name_for(ctx.doc, body)
        return [BodyMesh(body, *item) for item in club_meshes(ctx.club, finish)]
    scale = 1.0 if rule is None else rule.radius_scale
    if part in TRUNK_PARTS:
        segments = [_trunk_segment(ctx, body, part, caps, scale)]
    else:
        segments = _limb_segments(part, caps, scale)
    return _segment_meshes(ctx, body, part, segments)


def body_meshes(
    spec: Mapping[str, Any],
    doc: AppearanceDocument,
    club: ClubAssembly | None = None,
) -> list[BodyMesh]:
    """Skin, garment, blend, shoe (and, with ``club``, club) meshes per body.

    Preconditions: ``doc`` names only library or document materials (an
    unknown name raises ``ValueError``); the spec has the landmarks
    ``cross_section.body_sections`` needs. Postconditions: deterministic;
    every mesh in :data:`SPEC_BODY_FRAME`; every material resolvable; bodies
    in visual-skeleton order. The head body is omitted while the parametric
    head is enabled.
    """
    library.check_references(doc)
    skeleton = derive_visual_skeleton(spec)
    shapes: dict[str, list[Shape]] = {}
    for shape in skeleton.shapes:
        shapes.setdefault(shape.body, []).append(shape)
    ctx = _Context(doc, body_sections(spec), shapes, club)
    by_body: dict[str, list[Capsule]] = {}
    for cap in skeleton.capsules:
        by_body.setdefault(cap.body, []).append(cap)
    out = [
        m for body, caps in by_body.items() for m in _meshes_for_body(ctx, body, caps)
    ]
    known = library.library_materials(doc)
    unknown = sorted({m.material_name for m in out} - set(known))
    if unknown:
        raise ValueError(f"Body meshes name unknown materials {unknown}")
    return out


def resolve_appearance(
    spec: Mapping[str, Any],
    doc: AppearanceDocument,
    club: ClubAssembly | None = None,
    head_mesh: MeshReference | None = None,
) -> ResolvedAppearance:
    """The engine-neutral resolution an emitter translates (see module doc)."""
    meshes = tuple(body_meshes(spec, doc, club))
    known = library.library_materials(doc)
    materials = {m.material_name: known[m.material_name] for m in meshes}
    return ResolvedAppearance(meshes, dict(sorted(materials.items())), head_mesh)
