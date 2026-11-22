"""Opt-in scene lighting of the MuJoCo export (#15150).

The default export keeps the scene's own studio lights plus the sun
(``_sun_light``). ``--lighting showpiece`` swaps them for the directional
lights, headlight, shadow map and ground tone of a settings file
(``showpiece_lighting_settings.json``; meanings under its ``sources``).
Display only: lights, ``<visual>``, the model extent and the ground material
are rewritten; bodies, joints, geoms and contact settings are not touched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET

from src.tools.native_viewer_export.backends._sun_light import add_sun, load_sun_light

SETTINGS_PATH = Path(__file__).with_name("showpiece_lighting_settings.json")
FLOOR_GEOM = "visual_floor"
LIGHTING_CHOICES = ("studio", "showpiece")  # --lighting; the first is the default
_HEADLIGHT_KEYS = ("ambient", "diffuse", "specular")
_PARTS = {  # settings file section -> the fields it holds
    "shadow": ("shadowsize", "shadowclip", "extent_m"),
    "ground": ("texture_gain", "checker_contrast", "reflectance"),
}


def _vector(values: Any, label: str, unit: bool = False) -> tuple[float, ...]:
    """Three finite numbers (in [0, 1] when ``unit``); ``ValueError`` otherwise."""
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be three numbers, got {values!r}") from exc
    if len(vector) != len("xyz") or not all(math.isfinite(v) for v in vector):
        raise ValueError(f"{label} must be three finite numbers, got {values!r}")
    if unit and not all(0.0 <= v <= 1.0 for v in vector):
        raise ValueError(f"{label} must be three numbers in [0, 1], got {values!r}")
    return vector


@dataclass(frozen=True)
class SceneLight:
    """One directional light: where it shines and its linear RGB terms."""

    name: str
    direction: tuple[float, ...]
    diffuse: tuple[float, ...]
    specular: tuple[float, ...]
    castshadow: bool

    def __post_init__(self) -> None:
        if not (isinstance(self.name, str) and self.name):
            raise ValueError("a light needs a name")
        direction = _vector(self.direction, f"light {self.name} direction")
        if not direction[2] < 0.0:
            raise ValueError(f"light {self.name} direction must point down (z < 0)")
        if not isinstance(self.castshadow, bool):
            raise ValueError(f"light {self.name} castshadow must be true or false")
        object.__setattr__(self, "direction", direction)
        for term in ("diffuse", "specular"):
            colour = _vector(getattr(self, term), f"light {self.name} {term}", True)
            object.__setattr__(self, term, colour)


@dataclass(frozen=True)
class LightingSettings:
    """Validated lighting settings file (``ValueError`` when invalid)."""

    lights: tuple[SceneLight, ...]
    headlight: Mapping[str, tuple[float, ...]]
    shadowsize: int
    shadowclip: float
    extent_m: float
    texture_gain: float
    checker_contrast: float
    reflectance: float

    def __post_init__(self) -> None:
        names = [light.name for light in self.lights]
        if not names:
            raise ValueError("lights needs at least one light")
        if len(set(names)) != len(names):
            raise ValueError(f"light names must be unique, got {names}")
        if set(self.headlight) != set(_HEADLIGHT_KEYS):
            raise ValueError(f"headlight needs exactly {_HEADLIGHT_KEYS}")
        size = self.shadowsize
        if not isinstance(size, int) or size <= 0 or size & (size - 1):
            raise ValueError(f"shadowsize must be a power of two, got {size}")
        for key in ("shadowclip", "extent_m", "texture_gain"):
            if not 0.0 < float(getattr(self, key)) < math.inf:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("checker_contrast", "reflectance"):
            if not 0.0 <= float(getattr(self, key)) <= 1.0:
                raise ValueError(f"ground {key} must be in [0, 1]")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LightingSettings:
        """Settings from the file's mapping; a missing key is a ``ValueError``."""
        try:
            head = {
                key: _vector(value, f"headlight {key}", True)
                for key, value in raw["headlight"].items()
            }
            lights = tuple(SceneLight(**entry) for entry in raw["lights"])
            parts = {k: raw[part][k] for part, keys in _PARTS.items() for k in keys}
            return cls(lights=lights, headlight=head, **parts)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"lighting settings are incomplete: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """The mapping :meth:`from_dict` reads (what a worker job carries)."""
        flat = asdict(self)
        parts = {part: {k: flat[k] for k in keys} for part, keys in _PARTS.items()}
        return {"lights": flat["lights"], "headlight": flat["headlight"], **parts}


def load_lighting(path: Path | None = None) -> LightingSettings:
    """Read and validate a lighting settings file (default: the showpiece)."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return LightingSettings.from_dict(raw)


def _numbers(values: Any) -> str:
    return " ".join(format(float(v), ".6g") for v in values)


def _child(parent: Any, tag: str) -> Any:
    """The ``tag`` child of ``parent``, made when it is not there."""
    found = parent.find(tag)
    if found is None:
        found = parent.makeelement(tag, {})
        parent.append(found)
    return found


def _tone_ground(root: Any, settings: LightingSettings) -> None:
    """Scale the ground texture's checker colours and set its reflectance."""
    floor = root.find(f".//geom[@name='{FLOOR_GEOM}']")
    name = floor.get("material") if floor is not None else None
    material = root.find(f".//material[@name='{name}']")
    texture = material.get("texture") if material is not None else None
    tex = root.find(f".//texture[@name='{texture}']")
    if material is None or tex is None or not (tex.get("rgb1") and tex.get("rgb2")):
        raise ValueError(f"scene has no {FLOOR_GEOM} with a two-colour texture")
    material.set("reflectance", _numbers([settings.reflectance]))
    pair = [[float(v) for v in tex.get(key).split()] for key in ("rgb1", "rgb2")]
    mean = [sum(both) / len(pair) for both in zip(*pair, strict=True)]
    keep, gain = settings.checker_contrast, settings.texture_gain
    for key, colour in zip(("rgb1", "rgb2"), pair, strict=True):
        toned = [(m + (c - m) * keep) * gain for m, c in zip(mean, colour, strict=True)]
        tex.set(key, _numbers(min(1.0, value) for value in toned))


def apply_lighting(xml: str, settings: LightingSettings) -> str:
    """``xml`` with the scene lights, headlight, shadow map and ground of ``settings``.

    The lights that hang in ``<worldbody>`` itself are replaced. ``ValueError``
    without a ``<worldbody>`` or a textured ground plane.
    """
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("model has no <worldbody> to hang the lights in")
    for old in world.findall("light"):
        world.remove(old)
    for index, light in enumerate(settings.lights):
        attributes = {
            "name": light.name,
            "directional": "true",
            "dir": _numbers(light.direction),
            "diffuse": _numbers(light.diffuse),
            "specular": _numbers(light.specular),
            "castshadow": str(light.castshadow).lower(),
        }
        world.insert(index, world.makeelement("light", attributes))
    visual = _child(root, "visual")
    for key, value in settings.headlight.items():
        _child(visual, "headlight").set(key, _numbers(value))
    _child(visual, "quality").set("shadowsize", str(settings.shadowsize))
    _child(visual, "map").set("shadowclip", _numbers([settings.shadowclip]))
    _child(root, "statistic").set("extent", _numbers([settings.extent_m]))
    _tone_ground(root, settings)
    return ET.tostring(root, encoding="unicode")


def light_scene(xml: str, lighting: Mapping[str, Any] | None) -> str:
    """Scene of a worker job: the studio scene with the sun, or ``lighting``."""
    if lighting is None:
        return add_sun(xml, load_sun_light())
    return apply_lighting(xml, LightingSettings.from_dict(lighting))
