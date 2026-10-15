"""Named material library, presets and body-part classification.

Materials are referenced by name. Skin tones, clothing presets and club
finishes are tables of names; a document may add or override materials.
Body names in specs are heterogeneous (``femur_r``, ``solid_reference:...``),
so a body is classified to an anatomical *part* by case-insensitive pattern,
then the part is mapped to a material through the clothing preset.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

from src.shared.python.model_appearance.schema import (
    AppearanceDocument,
    Material,
    SegmentRule,
    Texture,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _skin(rgb: tuple[float, float, float]) -> Material:
    return Material(
        (*rgb, 1.0),
        roughness=0.55,
        texture=Texture("noise", noise=0.02, repeat=1.0),
    )


MATERIALS: dict[str, Material] = {
    "skin_light": _skin((0.93, 0.76, 0.66)),
    "skin_medium": _skin((0.80, 0.60, 0.46)),
    "skin_tan": _skin((0.66, 0.45, 0.32)),
    "skin_dark": _skin((0.40, 0.27, 0.20)),
    "polo_navy": Material(
        (0.16, 0.27, 0.55, 1.0), 0.85, texture=Texture("noise", noise=0.04)
    ),
    "polo_white": Material(
        (0.93, 0.93, 0.92, 1.0), 0.85, texture=Texture("noise", noise=0.04)
    ),
    "polo_red": Material(
        (0.70, 0.12, 0.14, 1.0), 0.85, texture=Texture("noise", noise=0.04)
    ),
    "shorts_khaki": Material(
        (0.78, 0.70, 0.52, 1.0), 0.9, texture=Texture("noise", noise=0.05)
    ),
    "trousers_charcoal": Material(
        (0.30, 0.32, 0.35, 1.0), 0.9, texture=Texture("noise", noise=0.05)
    ),
    "hair_brown": Material(
        (0.20, 0.13, 0.08, 1.0), 0.8, texture=Texture("noise", noise=0.05)
    ),
    "hair_black": Material(
        (0.05, 0.045, 0.045, 1.0), 0.75, texture=Texture("noise", noise=0.04)
    ),
    "hair_blond": Material(
        (0.70, 0.55, 0.30, 1.0), 0.8, texture=Texture("noise", noise=0.05)
    ),
    "cap_white": Material(
        (0.94, 0.94, 0.93, 1.0), 0.85, texture=Texture("noise", noise=0.04)
    ),
    "cap_navy": Material(
        (0.12, 0.20, 0.45, 1.0), 0.85, texture=Texture("noise", noise=0.04)
    ),
    "eye_white": Material((0.96, 0.96, 0.94, 1.0), 0.2),
    "iris_dark": Material((0.10, 0.07, 0.05, 1.0), 0.15),
    "brow_dark": Material((0.14, 0.09, 0.06, 1.0), 0.9),
    "lip_pink": Material((0.62, 0.30, 0.30, 1.0), 0.5),
    "shoe_white": Material((0.92, 0.92, 0.90, 1.0), 0.4),
    "shoe_black": Material((0.06, 0.06, 0.07, 1.0), 0.35),
    "glove_white": Material((0.95, 0.95, 0.94, 1.0), 0.7),
    "grip_rubber": Material((0.08, 0.08, 0.09, 1.0), 0.95),
    "satin_steel": Material((0.46, 0.48, 0.52, 1.0), 0.5, 1.0),
    "chrome": Material((0.86, 0.88, 0.92, 1.0), 0.08, 1.0),
    "graphite": Material((0.10, 0.11, 0.13, 1.0), 0.35, 0.6),
    "black_pvd": Material((0.05, 0.05, 0.06, 1.0), 0.25, 0.9),
    "turf": Material(
        (0.30, 0.50, 0.24, 1.0),
        0.95,
        texture=Texture(
            "checker",
            color_a=(0.27, 0.46, 0.21),
            color_b=(0.33, 0.54, 0.26),
            noise=0.05,
            repeat=8.0,
        ),
    ),
    "studio_floor": Material(
        (0.55, 0.57, 0.60, 1.0),
        0.6,
        texture=Texture(
            "checker",
            color_a=(0.50, 0.52, 0.55),
            color_b=(0.60, 0.62, 0.65),
            repeat=6.0,
        ),
    ),
}

SKIN_TONES = ("skin_light", "skin_medium", "skin_tan", "skin_dark")
HEADWEAR_DEFAULT_MATERIAL = {"hair": "hair_brown", "cap": "cap_navy"}
CLUB_FINISHES = ("satin_steel", "chrome", "graphite", "black_pvd")

# The lower-body garment is keyed on the thigh; the pelvis always wears it.
PANTS_PART = "thigh"


def _outfit(top: str, pants: str, shoes: str, *, full_length: bool) -> dict[str, str]:
    """One outfit: the shirt dresses the trunk and upper arms, the pants (and
    nothing else) dress the pelvis and thighs, and the shins when full length."""
    parts = {
        "torso": top,
        "shoulder": top,
        "upper_arm": top,
        "pelvis": pants,
        PANTS_PART: pants,
        "foot": shoes,
        "hand": "glove_white",
    }
    if full_length:
        parts["shin"] = pants
    return parts


# Anatomical part -> material, per clothing preset; absent parts show skin.
CLOTHING: dict[str, dict[str, str]] = {
    "none": {"foot": "shoe_white", "hand": "glove_white"},
    "golf_polo_shorts": _outfit(
        "polo_navy", "shorts_khaki", "shoe_white", full_length=False
    ),
    "golf_polo_trousers": _outfit(
        "polo_white", "trousers_charcoal", "shoe_black", full_length=True
    ),
}


def pants_material(preset: str) -> str | None:
    """Lower-body garment of ``preset`` (what the pelvis wears), or ``None``."""
    if preset not in CLOTHING:
        raise ValueError(f"Unknown clothing preset: {preset}")
    return CLOTHING[preset].get(PANTS_PART)


def check_clothing(presets: Mapping[str, Mapping[str, str]]) -> None:
    """Every preset names library materials, and its pelvis wears its pants."""
    for preset, parts in presets.items():
        for part, name in parts.items():
            if name not in MATERIALS:
                raise ValueError(
                    f"Unknown material name for clothing {preset}.{part}: {name}"
                )
        pants = parts.get(PANTS_PART)
        if parts.get("pelvis") != pants:
            raise ValueError(
                f"Clothing {preset}: pelvis garment {parts.get('pelvis')!r} must "
                f"equal the pants ({PANTS_PART}) garment {pants!r}"
            )


# One classification table: lowercased fnmatch patterns over ONE path
# component of the body name. A pattern without wildcards is an exact segment
# name (most specific), a trailing-only wildcard anchors a prefix, anything
# else is a substring match (least specific). Order is irrelevant.
PART_RULES: tuple[tuple[str, str], ...] = (
    # GolfSwing3D trunk segments sit in the "Hips and Torso Inputs" folder,
    # which names both the pelvis and the torso, so they are named exactly.
    ("lowertorso", "pelvis"),
    ("uppertorsobase", "torso"),
    ("comrod", "torso"),
    ("*clubface*", "club"),
    ("*club*", "club"),
    ("*shaft*", "club"),
    ("*grip*", "hand"),
    ("*hand*", "hand"),
    ("*upperarm*", "upper_arm"),
    ("*humerus*", "upper_arm"),
    ("*forearm*", "forearm"),
    ("*elbow*", "forearm"),
    ("*radius*", "forearm"),
    ("*ulna*", "forearm"),
    ("*hubto*", "shoulder"),
    ("*torso*", "torso"),
    ("*spine*", "torso"),
    ("*pelvis*", "pelvis"),
    ("*hips*", "pelvis"),
    ("femur*", "thigh"),
    ("*thigh*", "thigh"),
    ("tibia*", "shin"),
    ("*shank*", "shin"),
    ("talus*", "foot"),
    ("calcn*", "foot"),
    ("toes*", "foot"),
    ("*foot*", "foot"),
    ("*head*", "head"),
    ("*neck*", "head"),
)

# Visual-only cross-section radius per part (metres). Capsule radii from the
# shared skeleton are mass-derived and clamped thin; a character needs a torso
# wider than a limb. Appearance never feeds back into inertias.
PART_RADIUS_M: dict[str, float] = {
    "torso": 0.145,
    "pelvis": 0.125,
    "upper_arm": 0.043,
    "shoulder": 0.05,
    "forearm": 0.036,
    "hand": 0.042,
    "thigh": 0.078,
    "shin": 0.052,
    "foot": 0.040,
    "head": 0.095,
}
# Depth/width aspect of the cross-section (1 = round).
PART_ASPECT: dict[str, float] = {"torso": 0.72, "pelvis": 0.78, "foot": 0.7}
GARMENT_THICKNESS = 1.08
# Fractional extent of a garment along the segment: (start, end).
GARMENT_COVERAGE: dict[str, tuple[float, float]] = {
    "upper_arm": (0.0, 0.62),
    "thigh": (0.0, 0.6),
    "shin": (0.0, 1.0),
}
# Segment smoothness: loft resolution, and ball blends where limbs meet the trunk.
LOFT_RINGS, LOFT_SIDES = 22, 36
BLEND_PARTS = ("upper_arm", "thigh")
BLEND_RADIUS_SCALE = 1.0  # ball only fills the seam; never wider than the limb


def blend_scale(garment: bool) -> float:
    """Blend-ball radius over the limb radius (<= 1.1 incl. garment thickness)."""
    return BLEND_RADIUS_SCALE * (GARMENT_THICKNESS if garment else 1.0)


GARMENT_PARTS = ("torso", "pelvis", "shoulder", "upper_arm", "thigh", "shin")


def _specificity(pattern: str) -> int:
    """0 = exact segment name, 1 = prefix, 2 = substring (lower is more specific)."""
    if not any(c in pattern for c in "*?["):
        return 0
    return 1 if pattern.endswith("*") and not pattern[:-1].count("*") else 2


def _component_part(component: str, body_name: str) -> str | None:
    """Part of one path component from its most specific matching rules."""
    hits = [
        (_specificity(pattern), part)
        for pattern, part in PART_RULES
        if fnmatchcase(component, pattern)
    ]
    if not hits:
        return None
    best = min(rank for rank, _ in hits)
    parts = sorted({part for rank, part in hits if rank == best})
    if len(parts) > 1:
        raise ValueError(
            f"Ambiguous body classification for {body_name!r}: "
            f"{component!r} matches {parts} at equal specificity"
        )
    return parts[0]


def classify_body(body_name: str) -> str:
    """Anatomical part of a body, or ``"other"``.

    Specific before general: the leaf segment (text after the last ``/``)
    decides; only when no rule matches it does the nearest enclosing folder
    decide. The result never depends on the order of ``PART_RULES``.
    """
    if not isinstance(body_name, str) or not body_name:
        raise ValueError("Body name must be a non-empty string")
    for component in reversed(body_name.lower().split("/")):
        part = _component_part(component.strip(), body_name)
        if part is not None:
            return part
    return "other"


def library_materials(doc: AppearanceDocument | None = None) -> dict[str, Material]:
    """Built-in library overlaid with the document's own materials."""
    merged = dict(MATERIALS)
    if doc is not None:
        merged.update(doc.materials)
    return merged


def headwear_material_name(doc: AppearanceDocument) -> str | None:
    """Material of the hair or cap, or ``None`` when the head is bare."""
    if doc.head.headwear == "none":
        return None
    return doc.head.headwear_material or HEADWEAR_DEFAULT_MATERIAL[doc.head.headwear]


def check_references(doc: AppearanceDocument) -> None:
    """Every referenced material and preset name must exist."""
    materials = library_materials(doc)
    if doc.clothing not in CLOTHING:
        raise ValueError(f"Unknown clothing preset: {doc.clothing}")
    check_clothing({doc.clothing: CLOTHING[doc.clothing]})
    for label, name in (
        ("skin_tone", doc.skin_tone),
        ("club_finish", doc.club_finish),
        ("ground_material", doc.environment.ground_material),
        ("head headwear_material", headwear_material_name(doc) or "skin_light"),
        *(
            (f"segments[{i}]", r.material)
            for i, r in enumerate(doc.segments)
            if r.material
        ),
        *((f"clothing {p}", m) for p, m in CLOTHING[doc.clothing].items()),
    ):
        if name not in materials:
            raise ValueError(f"Unknown material name for {label}: {name}")


def rule_for_body(doc: AppearanceDocument, body_name: str) -> SegmentRule | None:
    lowered = body_name.lower()
    for rule in doc.segments:
        if fnmatchcase(lowered, rule.match.lower()):
            return rule
    return None


def material_name_for(doc: AppearanceDocument, body_name: str) -> str:
    """Name of the body's base (skin or equipment) material."""
    rule = rule_for_body(doc, body_name)
    if rule is not None and rule.material is not None:
        return rule.material
    part = classify_body(body_name)
    if part == "club":
        return doc.club_finish
    return doc.skin_tone  # clothing, gloves and shoes are garment layers


def garment_name_for(doc: AppearanceDocument, body_name: str) -> str | None:
    """Garment material draped over the body, or ``None`` when bare."""
    rule = rule_for_body(doc, body_name)
    if rule is not None and rule.material is not None:
        return None
    return CLOTHING[doc.clothing].get(classify_body(body_name))


def resolve_materials(
    doc: AppearanceDocument, body_names: Mapping[str, object] | list[str]
) -> dict[str, str]:
    """Body name -> base material name (postcondition: every name exists)."""
    materials = library_materials(doc)
    out = {}
    for name in body_names:
        chosen = material_name_for(doc, name)
        if chosen not in materials:
            raise ValueError(f"Unknown material for body {name}: {chosen}")
        out[name] = chosen
    return out
