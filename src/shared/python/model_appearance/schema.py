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
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "appearance-v1"
SCHEMA_PATH = Path(__file__).with_name("appearance_v1.schema.json")
APPEARANCE_SUFFIX = ".appearance.json"
# Spec keys that only ever affect pictures; excluded from the physics hash.
VISUAL_ONLY_SPEC_KEYS = ("visual_hints", "appearance")


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


@dataclass(frozen=True)
class AppearanceDocument:
    name: str = "default"
    skin_tone: str = "skin_medium"
    clothing: str = "golf_polo_shorts"
    club_finish: str = "satin_steel"
    materials: dict[str, Material] = field(default_factory=dict)
    segments: tuple[SegmentRule, ...] = ()
    environment: Environment = field(default_factory=Environment)
    spec_sha256: str | None = None


@lru_cache(maxsize=1)
def _validator() -> Any:
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
    texture = raw.get("texture")
    return Material(
        tuple(float(c) for c in raw["base_color"]),
        float(raw.get("roughness", 0.6)),
        float(raw.get("metallic", 0.0)),
        None if texture is None else _texture(texture),
    )


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
        skin_tone=data.get("skin_tone", "skin_medium"),
        clothing=data.get("clothing", "golf_polo_shorts"),
        club_finish=data.get("club_finish", "satin_steel"),
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
    )
    library.check_references(doc)
    return doc


def _texture_to_dict(texture: Texture) -> dict[str, Any]:
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
    }
    if doc.spec_sha256 is not None:
        out["spec_sha256"] = doc.spec_sha256
    validate_document(out)
    return out


def physics_spec_sha256(spec: bytes | Mapping[str, Any]) -> str:
    """Hash of a spec with all visual-only keys removed (canonical JSON).

    Postcondition: identical for two specs that differ only in
    ``visual_hints`` or an embedded ``appearance`` key.
    """
    data = json.loads(spec) if isinstance(spec, bytes | bytearray) else spec
    stripped = copy.deepcopy(dict(data))
    for key in VISUAL_ONLY_SPEC_KEYS:
        stripped.pop(key, None)
    canonical = json.dumps(stripped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def appearance_path_for(spec_path: str | Path) -> Path:
    """The appearance document that lives beside ``spec_path``."""
    path = Path(spec_path)
    return path.with_name(path.stem + APPEARANCE_SUFFIX)


def load_appearance(path: str | Path) -> AppearanceDocument:
    return document_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_appearance(doc: AppearanceDocument, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(document_to_dict(doc), indent=2) + "\n", encoding="utf-8"
    )
