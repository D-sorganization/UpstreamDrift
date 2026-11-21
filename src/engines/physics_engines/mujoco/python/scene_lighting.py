"""Lights and shadow quality of the MuJoCo visual scene (#15299).

Every number lives in ``scene_lighting.json`` beside this module. Appearance
only. A shadow map that covers too much ground draws saw-tooth fringes on lit
skin, a fill light with its own shadow doubles them, and lights that sum far
above one clip skin to white. MuJoCo sizes a directional light's map as
``shadowclip`` times the model extent and its depth bias grows with ``zfar``.
"""

import json
import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SETTINGS_PATH = Path(__file__).with_name("scene_lighting.json")
_SHARES = ("max_ragged_shadow_share", "max_clipped_share")
_MAP = ("haze", "znear", "zfar", "shadowclip", "shadowscale")


def _colour(values: Any, label: str) -> tuple[float, float, float]:
    """Three colour components in [0, 1]; raises ``ValueError`` otherwise."""
    colour = tuple(float(value) for value in values)
    if len(colour) != len("rgb") or not all(0.0 <= c <= 1.0 for c in colour):
        raise ValueError(f"{label} must be three numbers in [0, 1], got {values}")
    return colour  # type: ignore[return-value]


@dataclass(frozen=True)
class Light:
    """One scene light: horizontal offset and height above the ground point."""

    name: str
    offset_m: tuple[float, float]
    height_m: float
    diffuse: tuple[float, float, float]
    specular: tuple[float, float, float]
    castshadow: bool

    def __post_init__(self) -> None:
        if self.height_m <= 0.0:
            raise ValueError(f"{self.name} height_m must be positive")
        if not isinstance(self.castshadow, bool):
            raise ValueError(f"{self.name} castshadow must be true or false")


@dataclass(frozen=True)
class SceneLighting:
    """Validated contents of ``scene_lighting.json``."""

    plain: tuple[Light, ...]
    studio: tuple[Light, ...]
    headlight: Mapping[str, Mapping[str, tuple[float, float, float]]]
    shadowsize: int
    offsamples: int
    shadowclip: float
    shadowscale: float
    znear: float
    zfar: float
    haze: float
    acceptance: Mapping[str, Any]

    def __post_init__(self) -> None:
        size = self.shadowsize
        if size <= 0 or size & (size - 1):
            raise ValueError(f"shadowsize must be a power of two, got {size}")
        if self.offsamples < 0:
            raise ValueError("offsamples must not be negative")
        if self.shadowclip <= 0.0:
            raise ValueError("shadowclip must be positive")
        if not 0.0 < self.shadowscale <= 1.0:
            raise ValueError("shadowscale must be in (0, 1]")
        if not 0.0 < self.znear < self.zfar:
            raise ValueError("zfar must exceed a positive znear")
        if not 0.0 <= self.haze <= 1.0:
            raise ValueError("haze must be in [0, 1]")
        for key in _SHARES:
            if not 0.0 <= self.acceptance[key] <= 1.0:
                raise ValueError(f"{key} must be in [0, 1]")


def _rig(raw: Sequence[Mapping[str, Any]], label: str) -> tuple[Light, ...]:
    if not raw:
        raise ValueError(f"rig {label} needs at least one light")
    return tuple(
        Light(
            name=str(e["name"]),
            offset_m=(float(e["offset_m"][0]), float(e["offset_m"][1])),
            height_m=float(e["height_m"]),
            diffuse=_colour(e["diffuse"], f"{e['name']} diffuse"),
            specular=_colour(e["specular"], f"{e['name']} specular"),
            castshadow=e["castshadow"],
        )
        for e in raw
    )


def load_scene_lighting(path: Path | None = None) -> SceneLighting:
    """Read and validate the lighting settings; ``ValueError`` when invalid."""
    raw = json.loads((path or SETTINGS_PATH).read_text())
    rigs, quality, shadow_map = raw["rigs"], raw["quality"], raw["map"]
    headlight = {
        mode: {k: _colour(v, f"headlight {mode} {k}") for k, v in entry.items()}
        for mode, entry in raw["headlight"].items()
    }
    return SceneLighting(
        plain=_rig(rigs["plain"], "plain"),
        studio=_rig(rigs["studio"], "studio"),
        headlight=headlight,
        shadowsize=int(quality["shadowsize"]),
        offsamples=int(quality["offsamples"]),
        shadowclip=float(shadow_map["shadowclip"]),
        shadowscale=float(shadow_map["shadowscale"]),
        znear=float(shadow_map["znear"]),
        zfar=float(shadow_map["zfar"]),
        haze=float(shadow_map["haze"]),
        acceptance=raw["acceptance"],
    )


def _numbers(values: Any) -> str:
    return " ".join(format(float(v), ".6g") for v in np.asarray(values).ravel())


def add_lights(world: ET.Element, normal: np.ndarray, lights: Sequence[Light]) -> int:
    """Append ``lights``, aimed at the ground point; returns their count."""
    for light in lights:
        position = np.asarray(normal, dtype=float) * light.height_m
        position = position + np.array([*light.offset_m, 0.0])
        ET.SubElement(
            world,
            "light",
            name=f"visual_{light.name}",
            pos=_numbers(position),
            dir=_numbers(-position),
            diffuse=_numbers(light.diffuse),
            specular=_numbers(light.specular),
            castshadow=str(light.castshadow).lower(),
        )
    return len(lights)


def add_render_quality(visual: ET.Element, settings: SceneLighting, mode: str) -> None:
    """Append headlight, shadow quality and map elements to ``<visual>``.

    ``mode`` names a headlight entry of the settings (``studio`` or ``flat``).
    """
    if mode not in settings.headlight:
        raise ValueError(f"no headlight settings for lighting mode {mode!r}")
    head = {k: _numbers(v) for k, v in settings.headlight[mode].items()}
    ET.SubElement(visual, "headlight", head)
    quality = {k: str(getattr(settings, k)) for k in ("shadowsize", "offsamples")}
    ET.SubElement(visual, "quality", quality)
    ET.SubElement(visual, "map", {k: _numbers(getattr(settings, k)) for k in _MAP})
