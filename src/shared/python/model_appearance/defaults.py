"""The single default-golfer contract (HRG-B1, #13163).

The default skin tone, hair, clothing, club finish and the skin and hair
palettes live in ONE serialized file, ``src/config/appearance_defaults.json``.
Python reads it here and nowhere else; the web UI reads a TypeScript module
generated from the same file (``scripts/generate_appearance_defaults_ts.py``,
held fresh by a test), so no engine or panel carries its own copy.

Colour-space contract
---------------------
Every colour is authored as non-linear **sRGB**, components in 0..1.

* MuJoCo ``rgba`` takes the values directly (no conversion).
* three.js (the web UI) hands them to its own colour management
  (``THREE.ColorManagement`` with sRGB output), which linearises for lighting
  and encodes for display; the UI passes hex strings unchanged.
* No global linearisation is applied anywhere in the appearance layer.

The default golfer is a light-skinned (``skin_2``) golfer with mid brown hair
(``hair_brown``). Legacy names ``skin_light``/``skin_medium``/``skin_tan``/
``skin_dark`` remain as exact aliases of palette swatches so saved documents
still load unchanged.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.shared.python.model_appearance.schema import AppearanceDocument

CONTRACT_PATH = (
    Path(__file__).resolve().parents[4] / "src/config/appearance_defaults.json"
)
CONTRACT_VERSION = "appearance-defaults-v1"


@lru_cache(maxsize=1)
def _contract() -> dict[str, Any]:
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != CONTRACT_VERSION:
        raise ValueError(
            f"{CONTRACT_PATH} must declare schema_version {CONTRACT_VERSION!r}"
        )
    return data


def load_contract() -> dict[str, Any]:
    """The parsed contract (a fresh deep copy; callers may not mutate the cache)."""
    return json.loads(json.dumps(_contract()))


def default_fields() -> dict[str, str]:
    """Default document fields: skin_tone, headwear, headwear_material, ..."""
    return dict(_contract()["default"])


def skin_palette() -> list[dict[str, Any]]:
    """Skin swatches ``skin_1..skin_8``, light to dark (sRGB floats and hex)."""
    return load_contract()["skin_palette"]


def skin_aliases() -> dict[str, str]:
    """Legacy skin names (``skin_light`` ...) mapped to their palette swatch."""
    return dict(_contract()["skin_aliases"])


def hair_palette() -> list[dict[str, Any]]:
    """Hair colours, black to grey, with roughness and noise for the material."""
    return load_contract()["hair_palette"]


def default_document_dict() -> dict[str, Any]:
    """The default as an ``appearance-v1`` document mapping (validates)."""
    fields = default_fields()
    return {
        "schema_version": "appearance-v1",
        "skin_tone": fields["skin_tone"],
        "clothing": fields["clothing"],
        "club_finish": fields["club_finish"],
    }


def default_appearance(**overrides: Any) -> AppearanceDocument:
    """The default golfer document every engine and backend renders.

    ``overrides`` are merged over :func:`default_document_dict` (for example
    ``ball={"enabled": False}``). Postcondition: with no overrides the result
    equals ``document_from_dict({"schema_version": "appearance-v1"})``.
    """
    from src.shared.python.model_appearance.schema import document_from_dict

    return document_from_dict({**default_document_dict(), **overrides})
