"""Character appearance service: material library and validated sidecar.

CMB-7b (#11658): the API half of the web appearance/material picker. Builds
on the engine-agnostic appearance layer in
``src.shared.python.model_appearance`` (schema + library) without adding a
new module to that shared tree — a new file there would need a
divergence-inventory row, so this Qt-free service lives under
``src/api/services`` instead, alongside the other API services.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.schema import (
    HEADWEAR,
    SCHEMA_VERSION,
    AppearanceDocument,
    Environment,
    HeadSettings,
    appearance_path_for,
    document_from_dict,
    document_to_dict,
    material_to_dict,
)

# The two library materials built for ground/floor use. model_appearance has
# no existing "ground material" enumeration to reuse: it is derived here from
# the Environment default (`ground_material="turf"`, schema.py) plus its
# studio counterpart, rather than listing every material name (skin tones and
# club finishes are not sensible ground picks).
GROUND_MATERIALS: tuple[str, ...] = ("turf", "studio_floor")

# Request fields that map 1:1 onto AppearanceDocument top-level attributes.
_DOCUMENT_CHOICE_FIELDS = ("skin_tone", "clothing", "club_finish", "name")


def appearance_library() -> dict[str, Any]:
    """Pickable choices and every named material, for a web material picker.

    Postcondition: every name returned under ``skin_tones``, ``clothing``
    (values), ``club_finishes``, ``headwear_default_material`` (values) and
    ``ground_materials`` is a key of ``materials``.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "skin_tones": list(library.SKIN_TONES),
        "clothing": {preset: dict(parts) for preset, parts in library.CLOTHING.items()},
        "club_finishes": list(library.CLUB_FINISHES),
        "headwear": list(HEADWEAR),
        "headwear_default_material": dict(library.HEADWEAR_DEFAULT_MATERIAL),
        "ground_materials": list(GROUND_MATERIALS),
        "materials": {
            name: material_to_dict(material)
            for name, material in library.MATERIALS.items()
        },
    }


def _document_updates(choices: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: choices[field]
        for field in _DOCUMENT_CHOICE_FIELDS
        if choices.get(field) is not None
    }


def _head_with_choices(
    doc: AppearanceDocument, choices: Mapping[str, Any]
) -> HeadSettings:
    headwear = choices.get("headwear")
    headwear_material = choices.get("headwear_material")
    if headwear is None and headwear_material is None:
        return doc.head
    return replace(
        doc.head,
        headwear=doc.head.headwear if headwear is None else headwear,
        headwear_material=(
            doc.head.headwear_material
            if headwear_material is None
            else headwear_material
        ),
    )


def _environment_with_choices(
    doc: AppearanceDocument, choices: Mapping[str, Any]
) -> Environment:
    ground_material = choices.get("ground_material")
    if ground_material is None:
        return doc.environment
    return replace(doc.environment, ground_material=ground_material)


def build_appearance(
    choices: Mapping[str, Any], spec_sha256: str | None
) -> dict[str, Any]:
    """Apply picked library names to the default document and validate it.

    ``choices`` may set ``skin_tone``, ``clothing``, ``club_finish``,
    ``headwear``, ``headwear_material``, ``ground_material`` and ``name``;
    absent or ``None`` entries keep the ``AppearanceDocument`` default.
    ``spec_sha256``, when given, binds the appearance sidecar to the compiled
    physics spec it was picked for.

    Preconditions:
        ``choices`` must be a mapping; ``spec_sha256`` must be ``None`` or a
        string.

    Fails closed: an unknown skin tone, clothing preset, club finish,
    headwear material or ground material name raises ``ValueError`` (via
    ``document_to_dict``'s schema check and ``document_from_dict``'s library
    reference check) rather than silently falling back to a default.

    Postcondition: the returned dict round-trips through
    :func:`document_from_dict` (performed below as part of reference
    validation, not merely asserted).
    """
    if not isinstance(choices, Mapping):
        raise TypeError("choices must be a mapping")
    if spec_sha256 is not None and not isinstance(spec_sha256, str):
        raise TypeError("spec_sha256 must be a string or None")

    defaults = AppearanceDocument()
    doc = replace(
        defaults,
        **_document_updates(choices),
        head=_head_with_choices(defaults, choices),
        environment=_environment_with_choices(defaults, choices),
        spec_sha256=spec_sha256,
    )

    data = document_to_dict(doc)
    document_from_dict(data)
    return data


def appearance_sidecar_filename(
    document: Mapping[str, Any],
    *,
    preset_id: str | None,
    spec_sha256: str | None,
) -> str:
    """Canonical sidecar filename (schema.py's ``appearance_path_for``).

    Without a compiled spec, the stem is the preset id or the document name
    (``<stem>.appearance.json``). Bound to a compiled spec, the spec hash's
    first 8 hex characters are appended (``<stem>_<sha8>.appearance.json``) so
    two characters sharing a stem never collide — the same convention
    ``spec_export.export_character`` uses for engine exports.
    """
    stem = preset_id or str(document.get("name") or "character")
    if spec_sha256:
        stem = f"{stem}_{spec_sha256[:8]}"
    return appearance_path_for(stem).name


__all__ = [
    "GROUND_MATERIALS",
    "appearance_library",
    "appearance_sidecar_filename",
    "build_appearance",
]
