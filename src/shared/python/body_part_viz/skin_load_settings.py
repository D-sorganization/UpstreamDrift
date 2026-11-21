"""Settings of axial-load shading on a skinned body (#15150).

Read from ``skin_load_settings.json`` beside this module (see its ``sources``):
whether loads are blanked around ball impact, which body's load colours the
skin pieces of a body that has none of its own, and which materials keep
their own colour. Display only; no load is changed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SkinLoadSettings", "aliased_load_bodies", "load_skin_load_settings"]

SETTINGS_PATH = Path(__file__).with_name("skin_load_settings.json")


def _names(values: object, what: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise TypeError(f"{what} must be a list of names")
    names = tuple(values)
    if not all(isinstance(name, str) and name for name in names):
        raise TypeError(f"{what} must be non-empty strings")
    return tuple(str(name) for name in names)


@dataclass(frozen=True)
class SkinLoadSettings:
    """Validated ``skin_load_settings.json``."""

    axial_impact_blackout: bool
    load_body_alias: Mapping[str, str]
    unshaded_material_prefixes: tuple[str, ...]

    def __post_init__(self) -> None:
        """``TypeError`` for a wrong type; ``ValueError`` for a chained alias."""
        if not isinstance(self.axial_impact_blackout, bool):
            raise TypeError("axial_impact_blackout must be true or false")
        if not isinstance(self.load_body_alias, Mapping):
            raise TypeError("load_body_alias must map body names to body names")
        alias = dict(self.load_body_alias)
        _names(alias, "load_body_alias keys")
        _names(alias.values(), "load_body_alias values")
        if set(alias) & set(alias.values()):
            raise ValueError("load_body_alias must not chain: a value is also a key")
        prefixes = _names(self.unshaded_material_prefixes, "unshaded_material_prefixes")
        object.__setattr__(self, "load_body_alias", alias)
        object.__setattr__(self, "unshaded_material_prefixes", prefixes)


def load_skin_load_settings(path: Path | None = None) -> SkinLoadSettings:
    """Read the settings file; ``KeyError`` for a missing key, else as the class."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return SkinLoadSettings(
        raw["axial_impact_blackout"],
        raw["load_body_alias"],
        raw["unshaded_material_prefixes"],
    )


def aliased_load_bodies(
    piece_bodies: Mapping[str, str], alias: Mapping[str, str]
) -> dict[str, str]:
    """``piece_bodies`` (skin piece -> body) with each body replaced per ``alias``.

    The result is the piece -> load body mapping ``skin_piece_bindings`` takes:
    a piece of a body named in ``alias`` is coloured by the alias's load. An
    alias for a body no piece follows is ignored. Postcondition: same keys and
    order as ``piece_bodies``; neither argument is changed.
    """
    if not isinstance(piece_bodies, Mapping) or not isinstance(alias, Mapping):
        raise TypeError("piece_bodies and alias must be mappings")
    return {piece: alias.get(body, body) for piece, body in piece_bodies.items()}
