"""Engine-agnostic appearance document (``appearance-v1``).

An appearance document is stored *beside* a body/joint spec, never inside it:
per-part PBR materials (base colour, roughness, metallic, optional texture),
a skin tone, a clothing preset, a club finish and an environment. Materials
are referenced by name from the built-in library
(:mod:`src.shared.python.model_appearance.library`) or defined in the
document. Nothing here influences dynamics; :func:`physics_spec_sha256` hashes
a spec with every purely visual key removed so editing appearance, or the
visual hints, can never change the hash physics depends on.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from src.shared.python.core.canonical_floats import canonical_floats
from src.shared.python.core.handedness import (
    DEFAULT_HANDEDNESS,
    HANDEDNESS_CHOICES,
)
from src.shared.python.model_appearance.defaults import default_fields
from src.shared.python.model_appearance.head_morph import LEGACY_MORPH_FIELDS, HeadMorph

SCHEMA_VERSION = "appearance-v1"
SCHEMA_PATH = Path(__file__).with_name("appearance_v1.schema.json")
APPEARANCE_SUFFIX = ".appearance.json"
# Spec keys that only ever affect pictures; excluded from the physics hash.
VISUAL_ONLY_SPEC_KEYS = ("visual_hints", "appearance")
# Full-precision digests the physics hash leaves out (#14397): each would carry
# last-bit BLAS drift into it.  ``upper_body_sha256`` digests a slice of this
# same document, whose floats are hashed at canonical precision anyway; the
# parent's ``parent_spec_canonical_sha256`` (visual keys included) sits beside
# its ``parent_physics_spec_sha256``, which identifies the parent stably.
_FULL_PRECISION_DIGEST_PATHS = (
    ("upper_body_sha256",),
    ("spec_revision", "parent_spec_canonical_sha256"),
)
# Owner ruling (#12685): the default outfit is charcoal trousers. The preset
# itself is defined in ``library.CLOTHING``, which checks this name exists.
# Every default comes from the one contract (HRG-B1, #13163; see ``defaults``).
_DEFAULTS = default_fields()
DEFAULT_CLOTHING = _DEFAULTS["clothing"]
DEFAULT_SKIN_TONE = _DEFAULTS["skin_tone"]
DEFAULT_CLUB_FINISH = _DEFAULTS["club_finish"]
#: Which grip hands wear the golf glove, by golf role (HRG-G1, #13186). The
#: owner default is bare, skin-coloured hands; a document written before this
#: field existed resolves to ``"none"``.
GLOVE_CHOICES = ("none", "lead", "trail", "both")
DEFAULT_GLOVE = "none"
#: Maps the lead/trail roles to the anatomical side: the lead hand is the left
#: one for a right-handed golfer. The compiled spec carries no handedness, so
#: it is a document field that defaults to ``"right"``.


@dataclass(frozen=True)
class Texture:
    """A procedural (flat/checker/gradient/noise) or file-backed texture."""

    kind: str
    color_a: tuple[float, ...] | None = None
    color_b: tuple[float, ...] | None = None
    noise: float = 0.0
    repeat: float = 1.0
    path: str | None = None


@dataclass(frozen=True)
class Material:
    """Metallic-roughness PBR material with an optional texture."""

    base_color: tuple[float, ...]
    roughness: float = 0.6
    metallic: float = 0.0
    texture: Texture | None = None

    @property
    def rgba(self) -> tuple[float, float, float, float]:
        """Return the material's RGB colour with an opaque default alpha."""
        rgb = tuple(float(c) for c in self.base_color[:3])
        alpha = float(self.base_color[3]) if len(self.base_color) == 4 else 1.0
        return (rgb[0], rgb[1], rgb[2], alpha)


@dataclass(frozen=True)
class SegmentRule:
    """First matching rule (fnmatch, case-insensitive) styles a body."""

    match: str
    material: str | None = None
    mesh: str = "smooth"
    radius_scale: float = 1.0


@dataclass(frozen=True)
class Environment:
    ground_material: str = "turf"
    sky_top: tuple[float, ...] = (0.30, 0.50, 0.78)
    sky_bottom: tuple[float, ...] = (0.88, 0.92, 0.96)
    lighting: str = "studio"


BALL_SOURCES = ("address_geometry", "measured", "model_estimate")


@dataclass(frozen=True)
class BallSettings:
    """Decorative golf ball at address, visual only (GCV-13, #11719).

    ``position_m`` overrides the computed face-centre-plus-radius address
    placement, e.g. for a mocap-matched swing that holds a measured ball;
    ``source`` labels which it is and must be one of :data:`BALL_SOURCES`.
    """

    enabled: bool = True
    position_m: tuple[float, float, float] | None = None
    source: str = "address_geometry"

    def __post_init__(self) -> None:
        """Validate the source label and optional finite ball position."""
        if self.source not in BALL_SOURCES:
            raise ValueError(f"ball source must be one of {BALL_SOURCES}")
        if self.position_m is not None:
            pos = tuple(float(c) for c in self.position_m)
            if len(pos) != 3 or not all(math.isfinite(c) for c in pos):
                raise ValueError("ball position_m must be a finite 3-vector")


HEAD_AXES = ("+x", "-x", "+y", "-y", "+z", "-z")
HEADWEAR = ("none", "hair", "cap")
HEAD_MODELS = ("parametric", "cc0_mesh")
HAIR_STYLES = (
    "short_crop",
    "dome",
    "side_swept",
    "swept_back",
    "slick_part",
    "receding_crop",
)
HAIR_PARTS = ("left", "right", "none")
EYE_COLORS = ("blue_grey", "hazel", "brown")
CAP_STYLES = ("none", "baseball_cap", "flat_cap", "straw_hat", "newsboy_cap")
CAP_BRIM_RANGE = (0.10, 0.80)  # brim length, head lengths
DEFAULT_CAP_BRIM = 0.30
#: ``stature``: menton to vertex is ``height_over_stature`` of the subject's
#: stature (#15299); ``segment_length``: the spec's head segment length.
HEAD_SIZINGS = ("stature", "segment_length")
#: Adult male menton to top of head over stature: 232 mm of 1.756 m (Gordon et
#: al. 1989, 1988 US Army survey), a head of 1 : 7.6.
DEFAULT_HEAD_HEIGHT_OVER_STATURE = 0.132
HEAD_HEIGHT_OVER_STATURE_RANGE = (0.10, 0.16)
BODY_MODELS = ("ellipsoid", "meshes")
HEAD_OVERRIDE_FRAMES = ("parent", "world")


@dataclass(frozen=True)
class HeadOrientationOverride:
    """Visual-only head orientation channel (yaw, pitch, roll in radians).

    Declaring it detaches the visual head from the physics body so a gaze
    controller can drive the same stabilised head in every engine, including
    models without a neck joint. Head axes: x forward, y left, z up. Yaw is a
    left turn about +z, pitch looks up, roll tilts the head toward the right
    shoulder. ``frame`` says what the angles are relative to: ``parent`` (the
    head's own body frame) or ``world`` (a gaze-stabilised head). The values
    stored here are the static default; ``channel`` names the runtime feed.
    """

    frame: str = "parent"
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    channel: str = "gaze_head_ypr"


@dataclass(frozen=True)
class HeadSettings:
    """Visible head (skull, face features, neck, hair or cap), visual only."""

    enabled: bool = True
    headwear: str = "hair"
    headwear_material: str | None = None
    scale: float = 1.0
    forward_axis: str = "+x"
    up_axis: str = "+z"
    orientation_override: HeadOrientationOverride | None = None
    #: ``cc0_mesh`` (default, owner ruling #12949) or ``parametric`` (procedural).
    model: str = "cc0_mesh"
    #: ``short_crop`` (tapered sides, volume on top; CC0 head only) or the thin
    #: ``dome`` shell. The parametric head always draws the dome (#15276).
    hair_style: str = "short_crop"
    #: Side of the parting for ``short_crop``: ``left``, ``right`` or ``none``.
    hair_part: str = "left"
    #: Iris colour: ``blue_grey`` (default), ``hazel`` or ``brown``.
    eye_color: str = "blue_grey"
    #: ``none`` (default), ``baseball_cap`` or a low-profile ``flat_cap``.
    cap_style: str = "none"
    #: Library material of the cap (a colour); the brim shares it.
    cap_material: str = "cap_navy"
    #: Brim length in head lengths, within :data:`CAP_BRIM_RANGE`.
    cap_brim: float = DEFAULT_CAP_BRIM
    #: One of :data:`HEAD_SIZINGS`; ``segment_length`` is the size before #15299.
    sizing: str = "stature"
    #: Drawn menton to vertex over stature, in :data:`HEAD_HEIGHT_OVER_STATURE_RANGE`.
    height_over_stature: float = DEFAULT_HEAD_HEIGHT_OVER_STATURE
    #: Artistic face shape controls, in the canonical CC0 head frame.
    morph: HeadMorph = field(default_factory=HeadMorph)
    facial_hair: str = "none"

    def __post_init__(self) -> None:
        """Require a CC0 head whenever non-identity face morphs are set."""
        if not isinstance(self.morph, HeadMorph):
            raise ValueError("head morph must be a HeadMorph")
        if self.morph != HeadMorph() and self.model != "cc0_mesh":
            raise ValueError("head morph needs the cc0_mesh head model")
        if self.facial_hair not in ("none", "moustache"):
            raise ValueError("head facial_hair must be none or moustache")
        if self.facial_hair != "none" and self.model != "cc0_mesh":
            raise ValueError("facial hair needs the cc0_mesh head model")


@dataclass(frozen=True)
class FootSettings:
    """Foot and shoe model (HRG-F1, #13188), visual only."""

    #: ``procedural`` (shoe last, default) or ``cc0_mesh`` (reserved, unavailable).
    model: str = "procedural"


@dataclass(frozen=True)
class AppearanceDocument:
    name: str = "default"
    skin_tone: str = DEFAULT_SKIN_TONE
    clothing: str = DEFAULT_CLOTHING
    club_finish: str = DEFAULT_CLUB_FINISH
    materials: dict[str, Material] = field(default_factory=dict)
    segments: tuple[SegmentRule, ...] = ()
    environment: Environment = field(default_factory=Environment)
    spec_sha256: str | None = None
    head: HeadSettings = field(default_factory=HeadSettings)
    body_model: str = "ellipsoid"
    ball: BallSettings = field(default_factory=BallSettings)
    glove: str = DEFAULT_GLOVE
    handedness: str = DEFAULT_HANDEDNESS
    foot: FootSettings = field(default_factory=FootSettings)
    #: Visual grip picks (``style``, ``lead_thumb_offset_deg``); never a physical
    #: value (HRG-GR4, #13205).  ``None`` leaves the spec's own visual hints.
    grip: dict[str, Any] | None = None
    body_build: float = 0.0

    def __post_init__(self) -> None:
        """Keep the visual build control finite and within its supported range."""
        if (
            isinstance(self.body_build, bool)
            or not isinstance(self.body_build, (int, float))
            or not math.isfinite(self.body_build)
            or abs(self.body_build) > 1
        ):
            raise ValueError("body_build must be a finite number in [-1, 1]")


@lru_cache(maxsize=1)
def _validator() -> Any:
    """Build and cache the appearance-v1 JSON Schema validator."""
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def validate_document(data: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` listing every schema violation, else return."""
    if not isinstance(data, Mapping):
        raise TypeError("Appearance document must be a mapping")
    errors = sorted(
        _validator().iter_errors(dict(data)), key=lambda e: list(e.absolute_path)
    )
    if errors:
        detail = "; ".join(
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors
        )
        raise ValueError(f"Invalid appearance document: {detail}")


def _texture(raw: Mapping[str, Any]) -> Texture:
    """Parse a validated texture mapping into the immutable texture type."""
    if raw["kind"] == "file" and not raw.get("path"):
        raise ValueError("A file texture needs a path")
    a, b = raw.get("color_a"), raw.get("color_b")
    return Texture(
        kind=raw["kind"],
        color_a=None if a is None else tuple(float(c) for c in a),
        color_b=None if b is None else tuple(float(c) for c in b),
        noise=float(raw.get("noise", 0.0)),
        repeat=float(raw.get("repeat", 1.0)),
        path=raw.get("path"),
    )


def _material(raw: Mapping[str, Any]) -> Material:
    """Parse a validated material mapping and its optional texture."""
    texture = raw.get("texture")
    return Material(
        tuple(float(c) for c in raw["base_color"]),
        float(raw.get("roughness", 0.6)),
        float(raw.get("metallic", 0.0)),
        None if texture is None else _texture(texture),
    )


def _head_model(raw: Mapping[str, Any], defaults: HeadSettings) -> str:
    """Head model: explicit value, else the CC0 default (a cap needs parametric)."""
    if "model" in raw:
        return raw["model"]
    return "parametric" if raw.get("headwear") == "cap" else defaults.model


def _head_settings(raw: Mapping[str, Any]) -> HeadSettings:
    """Parse head controls, applying defaults and enforcing head constraints."""
    defaults = HeadSettings()
    ovr_raw = raw.get("orientation_override")
    override = None
    if ovr_raw is not None:
        d = HeadOrientationOverride()
        override = HeadOrientationOverride(
            ovr_raw.get("frame", d.frame),
            float(ovr_raw.get("yaw_rad", 0.0)),
            float(ovr_raw.get("pitch_rad", 0.0)),
            float(ovr_raw.get("roll_rad", 0.0)),
            ovr_raw.get("channel", d.channel),
        )
    if raw.get("forward_axis", "+x")[1] == raw.get("up_axis", "+z")[1]:
        raise ValueError("head forward_axis and up_axis must be different axes")
    choices = {
        "hair_style": HAIR_STYLES,
        "hair_part": HAIR_PARTS,
        "eye_color": EYE_COLORS,
        "cap_style": CAP_STYLES,
        "sizing": HEAD_SIZINGS,
    }
    picked = {k: raw.get(k, getattr(defaults, k)) for k in choices}
    for key, value in picked.items():
        if value not in choices[key]:
            raise ValueError(f"head {key} must be one of {choices[key]}, got {value!r}")
    brim = float(raw.get("cap_brim", defaults.cap_brim))
    if not CAP_BRIM_RANGE[0] <= brim <= CAP_BRIM_RANGE[1]:
        raise ValueError(f"head cap_brim must be within {CAP_BRIM_RANGE}, got {brim!r}")
    ratio = float(raw.get("height_over_stature", defaults.height_over_stature))
    low, high = HEAD_HEIGHT_OVER_STATURE_RANGE
    if not low <= ratio <= high:
        raise ValueError(f"head height_over_stature must be within {low} to {high}")
    model = _head_model(raw, defaults)
    if picked["cap_style"] != "none" and model != "cc0_mesh":
        raise ValueError("head cap_style needs the cc0_mesh head model")
    return HeadSettings(
        bool(raw.get("enabled", defaults.enabled)),
        raw.get("headwear", defaults.headwear),
        raw.get("headwear_material"),
        float(raw.get("scale", defaults.scale)),
        raw.get("forward_axis", defaults.forward_axis),
        raw.get("up_axis", defaults.up_axis),
        override,
        model,
        picked["hair_style"],
        picked["hair_part"],
        picked["eye_color"],
        picked["cap_style"],
        str(raw.get("cap_material", defaults.cap_material)),
        brim,
        picked["sizing"],
        ratio,
        HeadMorph(**raw.get("morph", {})),
        str(raw.get("facial_hair", "none")),
    )


def _head_to_dict(head: HeadSettings) -> dict[str, Any]:
    """Serialize non-default head controls into a compact mapping."""
    out: dict[str, Any] = {
        "enabled": head.enabled,
        "headwear": head.headwear,
        "scale": head.scale,
        "forward_axis": head.forward_axis,
        "up_axis": head.up_axis,
    }
    if head.model != HeadSettings().model:
        out["model"] = head.model
    for key in (
        "hair_style",
        "hair_part",
        "eye_color",
        "cap_style",
        "cap_material",
        "cap_brim",
        "sizing",
        "height_over_stature",
        "facial_hair",
    ):
        if getattr(head, key) != getattr(HeadSettings(), key):
            out[key] = getattr(head, key)
    if head.morph != HeadMorph():
        out["morph"] = {
            key: value
            for key, value in asdict(head.morph).items()
            if key in LEGACY_MORPH_FIELDS or value != 0
        }
    if head.headwear_material is not None:
        out["headwear_material"] = head.headwear_material
    ovr = head.orientation_override
    if ovr is not None:
        out["orientation_override"] = {
            "frame": ovr.frame,
            "yaw_rad": ovr.yaw_rad,
            "pitch_rad": ovr.pitch_rad,
            "roll_rad": ovr.roll_rad,
            "channel": ovr.channel,
        }
    return out


def _ball_settings(raw: Mapping[str, Any]) -> BallSettings:
    """Parse ball display settings and an optional position vector."""
    defaults = BallSettings()
    position = raw.get("position_m")
    return BallSettings(
        bool(raw.get("enabled", defaults.enabled)),
        None
        if position is None
        else cast("tuple[float, float, float]", tuple(float(c) for c in position)),
        raw.get("source", defaults.source),
    )


def _ball_to_dict(ball: BallSettings) -> dict[str, Any]:
    """Serialize ball display settings, including a measured position."""
    out: dict[str, Any] = {"enabled": ball.enabled, "source": ball.source}
    if ball.position_m is not None:
        out["position_m"] = list(ball.position_m)
    return out


def document_from_dict(data: Mapping[str, Any]) -> AppearanceDocument:
    """Validate and parse a document. Unknown library names are rejected."""
    from src.shared.python.model_appearance import library

    validate_document(data)
    materials = {k: _material(v) for k, v in data.get("materials", {}).items()}
    env_raw = data.get("environment", {})
    env_defaults = Environment()
    environment = Environment(
        env_raw.get("ground_material", env_defaults.ground_material),
        tuple(env_raw.get("sky_top", env_defaults.sky_top)),
        tuple(env_raw.get("sky_bottom", env_defaults.sky_bottom)),
        env_raw.get("lighting", env_defaults.lighting),
    )
    doc = AppearanceDocument(
        name=data.get("name", "default"),
        skin_tone=data.get("skin_tone", DEFAULT_SKIN_TONE),
        clothing=data.get("clothing", DEFAULT_CLOTHING),
        club_finish=data.get("club_finish", DEFAULT_CLUB_FINISH),
        materials=materials,
        segments=tuple(
            SegmentRule(
                r["match"],
                r.get("material"),
                r.get("mesh", "smooth"),
                float(r.get("radius_scale", 1.0)),
            )
            for r in data.get("segments", [])
        ),
        environment=environment,
        spec_sha256=data.get("spec_sha256"),
        head=_head_settings(data.get("head", {})),
        body_model=data.get("body_model", "ellipsoid"),
        body_build=data.get("body_build", 0.0),
        ball=_ball_settings(data.get("ball", {})),
        glove=data.get("glove", DEFAULT_GLOVE),
        handedness=data.get("handedness", DEFAULT_HANDEDNESS),
        foot=FootSettings(data.get("foot", {}).get("model", FootSettings().model)),
        grip=None if "grip" not in data else dict(data["grip"]),
    )
    library.check_references(doc)
    return doc


def _texture_to_dict(texture: Texture) -> dict[str, Any]:
    """Serialize a texture while omitting optional default values."""
    out: dict[str, Any] = {"kind": texture.kind}
    for key in ("color_a", "color_b"):
        value = getattr(texture, key)
        if value is not None:
            out[key] = list(value)
    if texture.noise:
        out["noise"] = texture.noise
    if texture.repeat != 1.0:
        out["repeat"] = texture.repeat
    if texture.path is not None:
        out["path"] = texture.path
    return out


def material_to_dict(material: Material) -> dict[str, Any]:
    """Serialize a material and its optional texture into schema fields."""
    out: dict[str, Any] = {
        "base_color": list(material.base_color),
        "roughness": material.roughness,
        "metallic": material.metallic,
    }
    if material.texture is not None:
        out["texture"] = _texture_to_dict(material.texture)
    return out


def document_to_dict(doc: AppearanceDocument) -> dict[str, Any]:
    """Serialise to a schema-valid dict (postcondition: round-trips)."""
    env = doc.environment
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": doc.name,
        "skin_tone": doc.skin_tone,
        "clothing": doc.clothing,
        "club_finish": doc.club_finish,
        "materials": {k: material_to_dict(m) for k, m in doc.materials.items()},
        "segments": [
            {
                "match": r.match,
                **({} if r.material is None else {"material": r.material}),
                "mesh": r.mesh,
                "radius_scale": r.radius_scale,
            }
            for r in doc.segments
        ],
        "environment": {
            "ground_material": env.ground_material,
            "sky_top": list(env.sky_top),
            "sky_bottom": list(env.sky_bottom),
            "lighting": env.lighting,
        },
        "head": _head_to_dict(doc.head),
        "body_model": doc.body_model,
        "ball": _ball_to_dict(doc.ball),
    }
    if doc.foot != FootSettings():
        out["foot"] = {"model": doc.foot.model}
    if doc.spec_sha256 is not None:
        out["spec_sha256"] = doc.spec_sha256
    if doc.glove != DEFAULT_GLOVE:  # defaults are omitted: old documents round-trip
        out["glove"] = doc.glove
    if doc.handedness != DEFAULT_HANDEDNESS:
        out["handedness"] = doc.handedness
    if doc.grip is not None:
        out["grip"] = dict(doc.grip)
    if doc.body_build:
        out["body_build"] = doc.body_build
    validate_document(out)
    return out


def physics_spec_sha256(spec: bytes | Mapping[str, Any]) -> str:
    """Hash of a spec with all visual-only keys removed (canonical JSON).

    Postcondition: identical for two specs that differ only in
    ``visual_hints`` or an embedded ``appearance`` key, and for two specs
    whose floats differ only below the canonical precision (12 significant
    digits, 1e-12 zero floor; :mod:`~src.shared.python.core.canonical_floats`,
    #14397), so BLAS or libm last-bit drift cannot move the hash.  The
    full-precision digests ``upper_body_sha256`` and
    ``spec_revision.parent_spec_canonical_sha256`` are left out for the same
    reason (see ``_FULL_PRECISION_DIGEST_PATHS``).
    """
    data = json.loads(spec) if isinstance(spec, bytes | bytearray) else spec
    stripped = copy.deepcopy(dict(data))
    for key in VISUAL_ONLY_SPEC_KEYS:
        stripped.pop(key, None)
    for *parents, key in _FULL_PRECISION_DIGEST_PATHS:
        node: Any = stripped
        for name in parents:
            node = node.get(name) if isinstance(node, Mapping) else None
        if isinstance(node, dict):
            node.pop(key, None)
    canonical = json.dumps(
        canonical_floats(stripped), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def appearance_path_for(spec_path: str | Path) -> Path:
    """The appearance document that lives beside ``spec_path``."""
    path = Path(spec_path)
    return path.with_name(path.stem + APPEARANCE_SUFFIX)


def load_appearance(path: str | Path) -> AppearanceDocument:
    """Load, validate, and parse an appearance sidecar from disk."""
    return document_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_appearance(doc: AppearanceDocument, path: str | Path) -> None:
    """Write a validated appearance sidecar as formatted JSON."""
    Path(path).write_text(
        json.dumps(document_to_dict(doc), indent=2) + "\n", encoding="utf-8"
    )
