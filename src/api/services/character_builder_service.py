"""Character builder service: parameters -> spec -> engine exports (CMB-3, #11654).

Pure functions behind ``src/api/routes/character_builder.py`` so the routes
stay thin and the logic is testable without HTTP. All compilation goes
through ``humanoid_character_builder.spec_params`` (CMB-1) and presets
through the shared presets loader (CMB-2).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.shared.python.humanoid_character_builder.presets.loader import (
    CharacterPreset,
    list_character_presets,
    load_character_preset,
)
from src.shared.python.humanoid_character_builder.spec_params import (
    SpecCharacterParameters,
    compile_full_body_spec,
    serialize_spec,
)

EXPORT_FORMATS: dict[str, tuple[str, str]] = {
    "spec": ("application/json", "json"),
    "urdf": ("text/xml", "urdf"),
    "mjcf": ("text/xml", "xml"),
    "osim": ("text/xml", "osim"),
}


@dataclass(frozen=True)
class CompiledCharacter:
    """A compiled character: parameters, spec document and its content hash."""

    parameters: SpecCharacterParameters
    document: dict[str, Any]
    sha256: str
    preset_id: str | None


def resolve_parameters(
    preset: str | None, overrides: Mapping[str, Any]
) -> tuple[SpecCharacterParameters, str | None]:
    """Return parameters from an optional preset plus non-None overrides.

    Raises ``ValueError`` for an unknown preset id or invalid values.
    """
    given = {k: v for k, v in overrides.items() if v is not None}
    if preset:
        loaded = load_character_preset(preset, **given)
        return loaded.parameters, loaded.id
    return SpecCharacterParameters.from_dict(given), None


def compile_character(
    preset: str | None, overrides: Mapping[str, Any]
) -> CompiledCharacter:
    """Resolve parameters and compile the full-body spec document."""
    params, preset_id = resolve_parameters(preset, overrides)
    document = compile_full_body_spec(params)
    sha = hashlib.sha256(serialize_spec(document).encode("utf-8")).hexdigest()
    return CompiledCharacter(params, document, sha, preset_id)


def _body_mass(body: Mapping[str, Any]) -> float:
    return float(sum(s["mass_kg"] for s in body["solids"]))


def build_summary(character: CompiledCharacter) -> dict[str, Any]:
    """Compact description of a compiled character for the build response."""
    doc = character.document
    return {
        "parameters": character.parameters.to_dict(),
        "preset": character.preset_id,
        "spec_sha256": character.sha256,
        "schema_version": doc["schema_version"],
        "qualification": doc["qualification"],
        "bodies": len(doc["bodies"]),
        "joints": len(doc["joints"]),
        "coordinates": len(doc["coordinate_order"]),
        "total_mass_kg": sum(_body_mass(b) for b in doc["bodies"]),
        "club": doc["club"]["name"],
    }


def build_preview(character: CompiledCharacter) -> dict[str, Any]:
    """Body masses and joint topology for a lightweight preview panel."""
    doc = character.document
    return {
        "spec_sha256": character.sha256,
        "stature_m": character.parameters.stature_m,
        "bodies": [
            {"name": b["name"], "mass_kg": _body_mass(b)} for b in doc["bodies"]
        ],
        "joints": [
            {"name": j["name"], "parent": j["parent"], "child": j["child"]}
            for j in doc["joints"]
        ],
    }


def export_character(character: CompiledCharacter, fmt: str) -> tuple[str, str, str]:
    """Return ``(text, media_type, filename)`` for ``fmt`` in ``EXPORT_FORMATS``.

    Engine exporters import lazily so listing presets never needs them.
    """
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format {fmt!r}; use {sorted(EXPORT_FORMATS)}")
    media_type, ext = EXPORT_FORMATS[fmt]
    text = serialize_spec(character.document)
    if fmt == "urdf":
        from src.engines.physics_engines.drake.python.full_body_urdf import (
            export_full_body_urdf,
        )

        text = export_full_body_urdf(text.encode("utf-8"))[0]
    elif fmt == "mjcf":
        from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
            export_full_body_mjcf,
        )

        text = export_full_body_mjcf(text.encode("utf-8"))[0]
    elif fmt == "osim":
        from src.engines.physics_engines.opensim.python.full_body_osim import (
            export_full_body_osim,
        )

        text = export_full_body_osim(character.document)[0]
    stem = character.preset_id or "custom"
    return text, media_type, f"{stem}_{character.sha256[:8]}.{ext}"


def preset_listing() -> list[dict[str, Any]]:
    """Metadata for every shipped preset, parameters included."""
    items: list[CharacterPreset] = [
        load_character_preset(pid) for pid in list_character_presets()
    ]
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "category": p.category,
            "parameters": p.parameters.to_dict(),
            "provenance": p.provenance,
            "limitations": p.limitations,
        }
        for p in items
    ]


__all__ = [
    "EXPORT_FORMATS",
    "CompiledCharacter",
    "build_preview",
    "build_summary",
    "compile_character",
    "export_character",
    "preset_listing",
    "resolve_parameters",
]
