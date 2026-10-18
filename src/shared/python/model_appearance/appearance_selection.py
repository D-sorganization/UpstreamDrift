"""Appearance selection: colour overrides and their precedence (HRG-B2, #13167).

One writer and one precedence rule shared by the ``native_viewer_export`` CLI,
the desktop character builder and (mirrored in ``appearanceDocument.ts``) the
web panel, so the same choice yields the same material values everywhere.

Precedence, highest first::

    explicit overrides (CLI flags / the user's picks)
      -> the selected document or ``<spec>.appearance.json`` sidecar
      -> the contract defaults (``defaults.default_document_dict``)

A remembered GUI selection (QSettings) may pre-fill a picker but is never an
input to :func:`resolve_appearance`; an export is reproducible from its
arguments alone.

Custom colours become entries of the document's ``materials`` map, authored as
non-linear sRGB in 0..1 exactly like the contract (``defaults`` docstring):

* skin: ``custom_skin`` (``skin_tone`` names it);
* hair: ``custom_hair`` (``head.headwear_material`` names it);
* top / bottom: the library garment material the clothing preset names for the
  torso / pants is *overridden by that name* (colour replaced, roughness and
  texture kept), because presets reference garments by library name.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.defaults import default_document_dict
from src.shared.python.model_appearance.schema import (
    GLOVE_CHOICES,
    Material,
    document_from_dict,
    document_to_dict,
    material_to_dict,
)

CUSTOM_SKIN = "custom_skin"
CUSTOM_HAIR = "custom_hair"
SKIN_REFERENCE = "skin_2"  # roughness and texture a custom skin colour inherits
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def is_hex(text: str) -> bool:
    """True when ``text`` is a ``#rrggbb`` colour."""
    return isinstance(text, str) and _HEX.match(text) is not None


def hex_to_rgb(text: str) -> tuple[float, float, float]:
    """``#rrggbb`` to sRGB floats in 0..1 (``byte / 255``, no rounding).

    Raises ``ValueError`` for anything but a six-digit hex colour.
    """
    if not is_hex(text):
        raise ValueError(f"colour must be a #rrggbb hex string, got {text!r}")
    return (
        int(text[1:3], 16) / 255.0,
        int(text[3:5], 16) / 255.0,
        int(text[5:7], 16) / 255.0,
    )


def rgb_to_hex(rgb: tuple[float, ...]) -> str:
    """sRGB floats in 0..1 to lowercase ``#rrggbb`` (channels clamped)."""
    return "#" + "".join(
        f"{round(min(1.0, max(0.0, float(c))) * 255):02x}" for c in rgb[:3]
    )


@dataclass(frozen=True)
class AppearanceOverrides:
    """Explicit appearance choices; ``None`` leaves a field to lower layers.

    ``skin_tone`` and ``hair`` take a library id or a ``#rrggbb`` colour;
    ``clothing`` is a preset id; ``top_color`` / ``bottom_color`` are hex;
    ``glove`` is ``none``, ``lead``, ``trail`` or ``both`` (golf roles; the
    document's ``handedness`` maps them to a side).
    """

    skin_tone: str | None = None
    hair: str | None = None
    clothing: str | None = None
    top_color: str | None = None
    bottom_color: str | None = None
    glove: str | None = None  # none | lead | trail | both (HRG-G1, #13186)

    def __post_init__(self) -> None:
        for name in ("top_color", "bottom_color"):
            value = getattr(self, name)
            if value is not None:
                hex_to_rgb(value)

    @property
    def is_empty(self) -> bool:
        return all(getattr(self, f) is None for f in self.__dataclass_fields__)


def _recoloured(base: Material, hex_colour: str) -> Material:
    """``base`` with its colour replaced; roughness, metal and texture kept."""
    return replace(base, base_color=(*hex_to_rgb(hex_colour), 1.0))


def _set_skin(doc: dict[str, Any], skin: str) -> None:
    if is_hex(skin):
        reference = library.MATERIALS.get(
            doc.get("skin_tone", ""), library.MATERIALS[SKIN_REFERENCE]
        )
        doc.setdefault("materials", {})[CUSTOM_SKIN] = material_to_dict(
            _recoloured(reference, skin)
        )
        doc["skin_tone"] = CUSTOM_SKIN
    else:
        doc["skin_tone"] = skin


def _set_hair(doc: dict[str, Any], hair: str) -> None:
    head = dict(doc.get("head", {}))
    headwear = head.get("headwear", "hair")
    if headwear != "hair":
        raise ValueError(
            f"a hair colour needs headwear 'hair', but the document has {headwear!r}"
        )
    if is_hex(hair):
        reference = library.MATERIALS[library.HEADWEAR_DEFAULT_MATERIAL["hair"]]
        doc.setdefault("materials", {})[CUSTOM_HAIR] = material_to_dict(
            _recoloured(reference, hair)
        )
        head["headwear_material"] = CUSTOM_HAIR
    else:
        head["headwear_material"] = hair
    doc["head"] = head


def _garment_name(preset: str, part: str, label: str) -> str:
    name = library.CLOTHING[preset].get(part)
    if name is None:
        raise ValueError(f"clothing preset {preset!r} has no {label} to recolour")
    return name


def _set_garment(doc: dict[str, Any], part: str, label: str, colour: str) -> None:
    name = _garment_name(doc["clothing"], part, label)
    doc.setdefault("materials", {})[name] = material_to_dict(
        _recoloured(library.MATERIALS[name], colour)
    )


def apply_overrides(
    document: Mapping[str, Any], overrides: AppearanceOverrides
) -> dict[str, Any]:
    """``document`` (appearance-v1 mapping) with ``overrides`` written in.

    The input is not mutated. Postcondition: the result is canonical
    (``document_to_dict`` of a validated document), so every unknown skin,
    hair, clothing or material name fails closed with ``ValueError``.
    """
    if not isinstance(overrides, AppearanceOverrides):
        raise TypeError("overrides must be an AppearanceOverrides")
    doc = _plain_copy(document)
    if overrides.clothing is not None:
        if overrides.clothing not in library.CLOTHING:
            raise ValueError(f"unknown clothing preset: {overrides.clothing!r}")
        doc["clothing"] = overrides.clothing
    doc.setdefault("clothing", default_document_dict()["clothing"])
    if overrides.glove is not None:
        if overrides.glove not in GLOVE_CHOICES:
            raise ValueError(
                f"glove must be one of {GLOVE_CHOICES}, got {overrides.glove!r}"
            )
        doc["glove"] = overrides.glove
    if overrides.skin_tone is not None:
        _set_skin(doc, overrides.skin_tone)
    if overrides.hair is not None:
        _set_hair(doc, overrides.hair)
    if overrides.top_color is not None:
        _set_garment(doc, "torso", "top", overrides.top_color)
    if overrides.bottom_color is not None:
        _set_garment(doc, library.PANTS_PART, "bottom", overrides.bottom_color)
    return document_to_dict(document_from_dict(doc))


def _plain_copy(document: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in document.items():
        out[key] = _plain_copy(value) if isinstance(value, Mapping) else value
    return out


def resolve_appearance(
    cli_overrides: AppearanceOverrides | None,
    document: Mapping[str, Any] | None,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The appearance an export renders: overrides, then document, then defaults.

    Pure: nothing is read from the environment, QSettings or the disk.
    ``document`` is a parsed appearance-v1 document or sidecar (or ``None``);
    ``defaults`` defaults to the contract. A document field replaces the
    default of the same name; an override replaces that field again.
    Postcondition: a canonical, validated appearance-v1 mapping.
    """
    base = _plain_copy(default_document_dict() if defaults is None else defaults)
    if document is not None:
        base.update(_plain_copy(document))
    resolved = apply_overrides(base, cli_overrides or AppearanceOverrides())
    return resolved


_LEGACY_FIELDS = {"skin": "skin_tone", "shirt": "top_color", "pants": "bottom_color"}


def _legacy_hex(key: str, value: Any) -> str:
    if (
        not isinstance(value, (list, tuple))
        or len(value) < 3
        or not all(isinstance(c, (int, float)) and 0.0 <= c <= 1.0 for c in value[:3])
    ):
        raise ValueError(
            f"legacy colors.{key} must be [r, g, b(, a)] floats in 0..1, got {value!r}"
        )
    return rgb_to_hex(tuple(value[:3]))


def overrides_from_legacy_config(config: Mapping[str, Any]) -> AppearanceOverrides:
    """Explicit one-time import of a legacy ``simulation_config.json`` (MuJoCo).

    Its ``colors`` block held RGBA floats for ``skin``, ``shirt`` and ``pants``;
    they become hex overrides (hair, shoes, eyes and club never mapped to the
    appearance contract and are left out). Precondition: a mapping; a malformed
    colour raises ``ValueError``. The result feeds :func:`resolve_appearance`.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    colors = config.get("colors") or {}
    return AppearanceOverrides(
        **{
            field: _legacy_hex(key, colors[key])
            for key, field in _LEGACY_FIELDS.items()
            if key in colors
        }
    )
