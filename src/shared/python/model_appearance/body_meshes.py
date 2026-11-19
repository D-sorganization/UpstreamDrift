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
    (``source="library_default"``). The chest's lower tip is its joint origin
    (the lumbar plane).

Pelvis rule (HRG-C1, #13164)
    A pelvis body that parents two hips and a torso joint is NOT lofted on a
    capsule (the longest one was the left hip after a hip calibration). It is
    a convex split ellipsoid centred on the HJC midpoint, vertical along the
    body's up axis, wide along the hip line: widest at the lowest HJC, running
    ``PELVIS_CROTCH_DROP_M`` below it and cut flat exactly at the lumbar plane.
    Half width = max(hint ML, HJC offset + femur radius + margin); half depth =
    the hint AP semi-axis. A pelvis without those landmarks keeps the loft.
    See
    ``docs/development/full_body_models/appearance_reference/shared_appearance_spec.tex``.

Head and club meshes stay in their own shared modules (``head``,
``club_assembly``/``grip_hands``); the head mesh reference slot (R7, #12687) names the
CC0 asset of ``model_appearance.head_mesh`` when ``head.model`` is ``cc0_mesh``. Visual only: nothing here touches mass, inertia or joints.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.model_appearance import foot, geometry, library
from src.shared.python.model_appearance import head_mesh as head_mesh_asset
from src.shared.python.model_appearance.club_assembly import (
    club_body_name,
    club_part_material,
    ClubAssembly,
)
from src.shared.python.model_appearance.club_chain_meshes import (
    ClubPart,
    club_meshes_by_body,
)
from src.shared.python.model_appearance.cross_section import (
    TRUNK_PARTS,
    BodySection,
    PelvisLandmarks,
    body_sections,
    pelvis_landmarks,
    width_hint_for,
)
from src.shared.python.model_appearance.grip_hands import assembly_meshes_with_hands
from src.shared.python.model_appearance.hand_spec import ResolvedHandSpec
from src.shared.python.model_appearance.pelvis_width import (
    PelvisWidth,
    resolve_pelvis_width,
)
from src.shared.python.model_appearance.schema import AppearanceDocument, Material
from src.shared.python.model_appearance.torso_profile import (
    TorsoFrame,
    TorsoProfile,
    TorsoStation,
    torso_mesh,
)
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
# The pelvis skin cap is recessed under the garment cap (no coplanar z-fighting).
_SKIN_CAP_INSET_M = 0.003
_MIN_YOKE_SINE = 0.5  # a shoulder link nearer to upright than this stays round


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
    club: ClubAssembly,
    finish: str,
    doc: AppearanceDocument,
    hands: ResolvedHandSpec | None = None,
) -> list[tuple[str, geometry.Mesh, str]]:
    """Shaft, grip, mesh head (GCV-11) and both grip hands (OSV-2), club frame.

    The hands are skin-coloured, or gloved per ``doc.glove`` (HRG-G1).
    """
    return [
        (label, mesh, club_part_material(label, finish, doc))
        for label, mesh in assembly_meshes_with_hands(club, hands=hands).items()
    ]


@dataclass(frozen=True)
class _Context:
    """Per-resolution inputs shared by every body."""

    doc: AppearanceDocument
    sections: Mapping[str, BodySection]
    pelvis: Mapping[str, PelvisLandmarks]
    shapes: Mapping[str, Sequence[Shape]]
    club: ClubAssembly | None
    #: Spec ``subject`` block (stature, optional measured hip breadth).
    subject: Mapping[str, Any]
    #: The parametric head draws the head body (skull, face, ears, neck).
    head_enabled: bool
    #: Procedural shoe-last shells of the foot bodies that could be sized (F1).
    feet: Mapping[str, Sequence[foot.FootPart]] = field(default_factory=dict)
    #: Club parts per body (``club_chain_meshes``; chain shaft R6, #12686).
    club_parts: Mapping[str, Sequence[ClubPart]] = field(default_factory=dict)
    #: Trunk bodies that carry a shoulder link: drawn with the thorax profile.
    thorax: frozenset[str] = frozenset()
    #: Parent body of each torso body that hangs from another torso body.
    parents: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _Segment:
    start: np.ndarray
    end: np.ndarray
    radius: float
    aspect: float
    end_scale: float = 1.0
    #: Full-radius axis of a flattened limb whose body has no landmark section.
    width_hint: np.ndarray | None = None
    #: Trunk cross-sections and the joint plane they are measured from (#15279).
    profile: TorsoProfile | None = None
    profile_start: np.ndarray | None = None


def _shoulder_segment(cap: Capsule, scale: float) -> _Segment:
    """The shoulder link as a yoke: flat, and starting beside the neck (#15279)."""
    start, end = np.asarray(cap.start_m), np.asarray(cap.end_m)
    along = end - start
    up = np.asarray(library.SHOULDER_UP_AXIS, dtype=float)
    radius = scale * library.PART_RADIUS_M["shoulder"]
    front = np.cross(up, along)
    if np.linalg.norm(front) < _MIN_YOKE_SINE * np.linalg.norm(along):
        # An upright link has no front-back axis from "up": keep it round.
        return _Segment(start, end, radius, 1.0)
    radius *= library.SHOULDER_RADIUS_SCALE
    droop = -library.SHOULDER_DROOP * radius * up  # the neck end hangs lower
    return _Segment(
        start + library.SHOULDER_START_FRACTION * along + droop,
        end,
        radius,
        library.SHOULDER_ASPECT,
        width_hint=front,
    )


def _limb_segments(part: str, caps: Sequence[Capsule], scale: float) -> list[_Segment]:
    out = []
    for cap in caps:
        if part == "shoulder":
            out.append(_shoulder_segment(cap, scale))
            continue
        radius = scale * (
            cap.radius_m
            if part in ("club", "other")
            else library.PART_RADIUS_M.get(part, cap.radius_m)
        )
        radius = min(radius, _MAX_RADIUS_OVER_LENGTH * cap.length_m())
        aspect = library.PART_ASPECT.get(part, 1.0)
        out.append(
            _Segment(
                np.asarray(cap.start_m),
                np.asarray(cap.end_m),
                radius,
                aspect,
                library.limb_distal_scale(part),
            )
        )
    return out


def _trunk_segment(
    ctx: _Context, body: str, part: str, caps: Sequence[Capsule], scale: float
) -> _Segment:
    """One trunk solid along the body's longest capsule (no overlapping lofts).

    The spindle's lower tip is the body's joint origin (the loft overshoots a
    segment by half a radius, so the segment starts that much above it): the
    chest then begins at the lumbar plane where the pelvis solid ends.
    """
    spine = max(caps, key=lambda c: c.length_m())  # first wins on a tie
    size = _section_size(part, ctx.shapes.get(body, ()), ctx.sections.get(body))
    radius = scale * size.radius_m
    if size.source == "library_default":
        radius = min(radius, _MAX_RADIUS_OVER_LENGTH * spine.length_m())
    start, end = np.asarray(spine.start_m), np.asarray(spine.end_m)
    axis = (end - start) / spine.length_m()
    over = min(0.5 * radius, 0.5 * spine.length_m())  # keep a positive length
    profile = None
    if size.source == "visual_hints" and part == "torso":
        role = "thorax" if body in ctx.thorax else "abdomen"
        profile = TorsoProfile(
            tuple(TorsoStation(*row) for row in library.TORSO_PROFILES[role]),
            library.TORSO_SQUARENESS,
        )
        if not ctx.shapes.get(ctx.parents.get(body, "")):
            # Nothing drawn as a trunk solid under it (the pelvis ends at this
            # joint plane): start there instead of tucking in.
            profile = profile.clipped(0.0)
    return _Segment(
        start + over * axis,
        end,
        radius,
        size.aspect,
        profile=profile,
        profile_start=start,
    )


@dataclass(frozen=True)
class _PelvisSegment(_Segment):
    """Pelvis solid: a split ellipsoid on the hip landmarks (see HRG-C1)."""

    shape: geometry.SplitEllipsoid | None = None


def _pelvis_plan(
    marks: PelvisLandmarks,
    section: BodySection,
    size: SectionSize,
    scale: float,
    subject: Mapping[str, Any],
) -> tuple[geometry.SplitEllipsoid, PelvisWidth]:
    """Hip-centred solid from the lower pole up to the lumbar plane (HRG-C4).

    Centre: the HJC midpoint, in the horizontal plane, at the lowest HJC
    level (the widest level). Width: ``pelvis_width.resolve_pelvis_width``
    (measured breadth or stature prior, raised for femoral-head containment).
    Half depth: the hint AP semi-axis. The lower pole is
    ``PELVIS_CROTCH_DROP_M`` below the *mean* HJC height, so vertical HJC
    asymmetry does not stretch it; the top is cut flat at the lumbar plane.
    """
    up, ml = section.trunk_up_axis(), section.ml_axis
    hjc = np.stack([marks.hjc_r, marks.hjc_l])
    z_hjc = hjc @ up
    z_low = float(z_hjc.min())
    top = float(marks.lumbar @ up - z_low)
    if top <= 0:
        raise ValueError(f"{marks.body}: the lumbar joint is not above the hip joints")
    half_up = top / math.sqrt(1.0 - library.PELVIS_WAIST_RATIO**2)
    mid = hjc.mean(axis=0)
    centre = mid - (mid @ up) * up + z_low * up
    width = resolve_pelvis_width(
        np.abs((hjc - mid) @ ml),
        np.clip((z_hjc - z_low) / half_up, 0.0, 0.999),
        hip_breadth_m=subject.get("hip_breadth_m"),
        stature_m=subject.get("stature_m"),
        hint_width_m=2.0 * size.radius_m,
    )
    pole = float(z_hjc.mean()) - library.PELVIS_CROTCH_DROP_M
    half_down = max(z_low - pole, library.PELVIS_MIN_BELOW_HJC_M)
    shape = geometry.SplitEllipsoid(
        centre,
        up,
        ml,
        scale * width.width_m / 2.0,
        scale * size.radius_m * size.aspect,
        half_up,
        half_down,
        top,
    )
    return shape, width


def _pelvis_shape(
    ctx: _Context, body: str, size: SectionSize, scale: float
) -> geometry.SplitEllipsoid:
    """The pelvis solid of ``body`` (see :func:`_pelvis_plan`)."""
    shape, _ = _pelvis_plan(
        ctx.pelvis[body], ctx.sections[body], size, scale, ctx.subject
    )
    return shape


def pelvis_width_report(spec: Mapping[str, Any]) -> dict[str, PelvisWidth]:
    """Resolved width, source and status (``containment_dominated`` ...) per pelvis.

    Postcondition: one entry per body with hip and lumbar landmarks.
    """
    sections = body_sections(spec)
    out: dict[str, PelvisWidth] = {}
    for body, marks in pelvis_landmarks(spec).items():
        if body not in sections:
            continue
        shapes = [s for s in derive_visual_skeleton(spec).shapes if s.body == body]
        size = _section_size(library.classify_body(body), shapes, sections[body])
        out[body] = _pelvis_plan(
            marks, sections[body], size, 1.0, spec.get("subject", {})
        )[1]
    return out


def _pelvis_segment(
    ctx: _Context, body: str, part: str, caps: Sequence[Capsule], scale: float
) -> _Segment:
    """The pelvis solid; bodies without hip and lumbar landmarks keep the loft."""
    if body not in ctx.pelvis or body not in ctx.sections:
        return _trunk_segment(ctx, body, part, caps, scale)
    size = _section_size(part, ctx.shapes.get(body, ()), ctx.sections[body])
    shape = _pelvis_shape(ctx, body, size, scale)
    centre = np.asarray(shape.centre)
    return _PelvisSegment(
        centre, centre + shape.top * shape.up, shape.half_width, 1.0, shape=shape
    )


def _loft(ctx: _Context, body: str, seg: _Segment, **extra: Any) -> geometry.Mesh:
    if isinstance(seg, _PelvisSegment) and seg.shape is not None:
        thickness = extra.get("thickness", 1.0)
        shape = seg.shape
        if thickness == 1.0:  # skin sits just under the garment's flat cap
            shape = dataclasses.replace(shape, top=shape.top - _SKIN_CAP_INSET_M)
        return geometry.split_ellipsoid_mesh(
            shape,
            thickness=thickness,
            rings=library.LOFT_RINGS // 2,
            sides=library.LOFT_SIDES,
        )
    hint = seg.width_hint
    if hint is None:
        hint = width_hint_for(ctx.sections, body, seg.start, seg.end)
    whole = extra.get("coverage", (0.0, 1.0)) == (0.0, 1.0)
    if seg.profile is not None and seg.profile_start is not None and whole:
        if hint is None:
            raise ValueError(f"torso body {body!r} has no width axis to loft on")
        frame = TorsoFrame(
            seg.profile_start,
            seg.end,
            np.asarray(hint, dtype=float),
            seg.radius,
            seg.radius * seg.aspect,
        )
        thickness = extra.get("thickness", 1.0)
        inset = _SKIN_CAP_INSET_M if thickness == 1.0 else 0.0
        return torso_mesh(frame, seg.profile, thickness=thickness, cap_inset_m=inset)
    return geometry.lofted_segment(
        seg.start,
        seg.end,
        seg.radius,
        geometry.LoftOptions(
            aspect=seg.aspect,
            end_scale=seg.end_scale,
            rings=library.LOFT_RINGS,
            sides=library.LOFT_SIDES,
            width_hint=hint,
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
            size = seg.radius * library.blend_scale(garment is not None, part)
            across, down = (
                library.DELTOID_HALF_SIZES if part == "upper_arm" else (1.0, 1.0)
            )
            along = (seg.end - seg.start) / np.linalg.norm(seg.end - seg.start)
            drop = library.DELTOID_DROP * seg.radius if part == "upper_arm" else 0.0
            ball = geometry.ellipsoid_mesh(
                seg.start + drop * along,
                size * np.array([down, across, across]),
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


def _foot_meshes(ctx: _Context, body: str) -> list[BodyMesh]:
    """Shoe (garment), bare-foot (skin) and ankle-stub shells of one foot body.

    A bare foot (no shoe garment) shows the foot shell only; the ``ankle`` stub
    takes the shin garment (trouser leg) when the outfit has one, else skin. It
    has its own label so "every foot garment is a shoe" stays true.
    """
    base = library.material_name_for(ctx.doc, body)
    garment = library.garment_name_for(ctx.doc, body)
    shin = library.CLOTHING[ctx.doc.clothing].get("shin")
    plan = (
        ("skin", "skin", base),
        ("garment", "garment", garment),
        ("sole", "sole", library.SOLE_MATERIAL if garment else None),
        ("ankle", "ankle", shin or base),
    )
    out: list[BodyMesh] = []
    counts: dict[str, int] = {}
    for role, label, material in plan:
        if material is None:
            continue
        for part in (p for p in ctx.feet[body] if p.role == role):
            i = counts[label] = counts.get(label, -1) + 1
            out.append(BodyMesh(body, f"{label}{i}", part.mesh, material))
    return out


def _meshes_for_body(
    ctx: _Context, body: str, caps: Sequence[Capsule]
) -> list[BodyMesh]:
    rule = library.rule_for_body(ctx.doc, body)
    if rule is not None and rule.mesh == "none":
        return []
    part = library.classify_body(body)
    if part == "head" and ctx.head_enabled:
        return []  # drawn by the shared head: skull, face, ears and neck
    if part == "hand" and ctx.club is not None and library.is_grip_hand_standoff(body):
        return []  # the grip hands are drawn on the club; no second hand loft
    if part == "club" and ctx.club is not None:
        return [BodyMesh(body, *item) for item in ctx.club_parts.get(body, ())]
    if body in ctx.feet:
        return _foot_meshes(ctx, body)
    scale = 1.0 if rule is None else rule.radius_scale
    if part == "pelvis":
        segments = [_pelvis_segment(ctx, body, part, caps, scale)]
    elif part in TRUNK_PARTS:
        segments = [_trunk_segment(ctx, body, part, caps, scale)]
    else:
        segments = _limb_segments(part, caps, scale)
    return _segment_meshes(ctx, body, part, segments)


def _club_parts(
    spec: Mapping[str, Any], doc: AppearanceDocument, club: ClubAssembly | None
) -> dict[str, list[ClubPart]]:
    """Club parts per body, in the club body's finish (empty without a club)."""
    body = club_body_name(spec)
    if club is None or body is None:
        return {}
    finish = library.material_name_for(doc, body)
    return club_meshes_by_body(spec, club, finish, doc)


def _parents_of(spec: Mapping[str, Any], part: str) -> frozenset[str]:
    """Bodies that carry a body of ``part`` (the thorax carries the shoulders)."""
    return frozenset(
        str(joint["parent"])
        for joint in spec.get("joints", ())
        if library.classify_body(str(joint.get("child", ""))) == part
    )


def _torso_parents(spec: Mapping[str, Any]) -> dict[str, str]:
    """Parent of every torso body whose parent is a torso body too."""
    return {
        str(joint["child"]): str(joint["parent"])
        for joint in spec.get("joints", ())
        if library.classify_body(str(joint.get("child", ""))) == "torso"
        and library.classify_body(str(joint.get("parent", ""))) == "torso"
    }


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
    sections = body_sections(spec)
    ctx = _Context(
        doc,
        sections,
        pelvis_landmarks(spec),
        shapes,
        club,
        spec.get("subject", {}),
        doc.head.enabled,
        foot.foot_parts(spec, sections, model=doc.foot.model),
        _club_parts(spec, doc, club),
        _parents_of(spec, "shoulder"),
        _torso_parents(spec),
    )
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
    if head_mesh is None and doc.head.enabled and doc.head.model == "cc0_mesh":
        head_mesh = MeshReference(*head_mesh_asset.mesh_reference())
    return ResolvedAppearance(meshes, dict(sorted(materials.items())), head_mesh)
