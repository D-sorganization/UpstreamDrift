"""Directional sun of the MuJoCo export scene (#15150).

The scene's own lights hang over the golfer and their cones end a few metres
out, so a wide ground read black towards the horizon. A directional light lights
level ground evenly at any distance. Numbers: ``sun_light_settings.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

SETTINGS_PATH = Path(__file__).with_name("sun_light_settings.json")
LIGHT_NAME = "export_sun"
_SKY_HEIGHT_M = 10.0  # a directional light's position only places its gizmo


@dataclass(frozen=True)
class SunLight:
    """Validated ``sun_light_settings.json`` (``ValueError``/``TypeError``)."""

    enabled: bool
    direction: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    castshadow: bool

    def __post_init__(self) -> None:
        if not (isinstance(self.enabled, bool) and isinstance(self.castshadow, bool)):
            raise TypeError("enabled and castshadow must be true or false")
        if len(self.direction) != 3 or not self.direction[2] < 0.0:
            raise ValueError("direction must be 3 numbers pointing down (z < 0)")
        if len(self.color) != 3 or not all(0.0 <= c <= 1.0 for c in self.color):
            raise ValueError("color must be 3 components in [0, 1]")
        if not (math.isfinite(self.intensity) and self.intensity >= 0.0):
            raise ValueError("intensity must be finite and not negative")

    @property
    def diffuse(self) -> tuple[float, float, float]:
        """Colour scaled by the intensity, clipped to MuJoCo's [0, 1]."""
        r, g, b = (min(1.0, c * self.intensity) for c in self.color)
        return r, g, b


def load_sun_light(path: Path | None = None) -> SunLight:
    """Read the settings file; ``KeyError`` for a missing key."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    keys = ("enabled", "direction", "color", "intensity", "castshadow")
    values = [raw[k] for k in keys]
    values[1], values[2] = tuple(values[1]), tuple(values[2])
    return SunLight(*values)


def _numbers(values: tuple[float, ...]) -> str:
    return " ".join(format(float(v), ".6g") for v in values)


def add_sun(xml: str, sun: SunLight) -> str:
    """``xml`` with the directional sun first in ``<worldbody>``.

    Unchanged when disabled; ``ValueError`` without a ``<worldbody>``. The light
    is named ``export_sun``, has no specular term and casts a shadow only when set.
    """
    if not sun.enabled:
        return xml
    tag = "<worldbody>"
    if tag not in xml:
        raise ValueError("model has no <worldbody> to hang the sun in")
    light = (
        f'<light name="{LIGHT_NAME}" directional="true" '
        f'pos="0 0 {_SKY_HEIGHT_M:g}" dir="{_numbers(sun.direction)}" '
        f'diffuse="{_numbers(sun.diffuse)}" specular="0 0 0" '
        f'castshadow="{str(sun.castshadow).lower()}"/>'
    )
    return xml.replace(tag, f"{tag}\n{light}", 1)
