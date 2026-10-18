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
import hashlib
from typing import Any

from src.shared.python.humanoid_character_builder import spec_export
from src.shared.python.humanoid_character_builder.spec_export import CompiledCharacter
from src.shared.python.humanoid_character_builder.spec_params import serialize_spec
from src.shared.python.motion_matching.execution.grip_selection import (
    GRIP_PRESET_CHOICES,
    GRIP_STYLE_CHOICES,
    PHYSICAL_PICKS,
    VISUAL_PICKS,
    GripOverrides,
    apply_grip_selection,
    apply_visual_picks,
    resolve_grip,
    visual_grip_block,
)
from src.shared.python.motion_matching.grip_contract import (
    GRIP_STRENGTH_PRESETS_DEG,
    NAMED_GRIP_PRESETS,
    grip_spec_to_physical,
)
from src.shared.python.model_appearance.schema import physics_spec_sha256

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.appearance_selection import (
    AppearanceOverrides,
    apply_overrides,
    hex_to_rgb,
)
from src.shared.python.model_appearance.schema import (
    GLOVE_CHOICES,
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


# Custom ``#rrggbb`` colour choices (HRG-B2, #13167) and the AppearanceOverrides
# field each one fills; they are written as document ``materials`` entries by the
# shared writer, so the CLI, desktop and web produce the same material values.
COLOUR_CHOICES: dict[str, str] = {
    "skin_color": "skin_tone",
    "hair_color": "hair",
    "top_color": "top_color",
    "bottom_color": "bottom_color",
}


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
        "glove": list(GLOVE_CHOICES),
        "grip": {
            "presets": dict(GRIP_STRENGTH_PRESETS_DEG),
            "named_presets": {
                name: {
                    "label": p.label,
                    "style": p.style.value,
                    "trigger_finger": p.trigger_finger,
                    "hand_spacing_m": p.hand_spacing_m,
                    "lead": {
                        "alpha_deg": p.lead.alpha_deg,
                        "psi_deg": p.lead.psi_deg,
                        "gamma_deg": p.lead.gamma_deg,
                    },
                    "trail": {
                        "alpha_deg": p.trail.alpha_deg,
                        "psi_deg": p.trail.psi_deg,
                        "gamma_deg": p.trail.gamma_deg,
                    },
                    "qualitative_source": p.qualitative_source,
                    "evidence_still_description": p.evidence_still_description,
                }
                for name, p in NAMED_GRIP_PRESETS.items()
            },
            "styles": list(GRIP_STYLE_CHOICES),
        },
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
    ``headwear``, ``headwear_material``, ``ground_material`` and ``name``,
    plus the custom ``#rrggbb`` colours ``skin_color``, ``hair_color``,
    ``top_color`` and ``bottom_color`` (written as document ``materials``) and
    the ``glove`` option (``none``, ``lead``, ``trail`` or ``both``);
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
    colours = {
        field: choices[choice]
        for choice, field in COLOUR_CHOICES.items()
        if choices.get(choice) is not None
    }
    for choice in COLOUR_CHOICES:
        if choices.get(choice) is not None:
            hex_to_rgb(choices[choice])  # ValueError unless #rrggbb
    if choices.get("glove") is not None:  # HRG-G1: an option, not a colour
        colours["glove"] = choices["glove"]
    if colours:
        data = apply_overrides(data, AppearanceOverrides(**colours))
    visual = {k: choices[k] for k in VISUAL_PICKS if choices.get(k) is not None}
    if visual:  # HRG-GR4: picture-only grip picks; physical ones never get here
        data = document_to_dict(
            document_from_dict(apply_visual_picks(data, GripOverrides(**visual)))
        )
    return data


def compile_with_grip(
    character: CompiledCharacter, overrides: GripOverrides
) -> CompiledCharacter:
    """``character`` with the physical grip picks as a new spec revision (GR4).

    Without a physical pick the character is returned as it is.  The new
    ``sha256`` is the revision's content hash, computed as ``compile_character``
    does; the revision needs a refit.
    """
    if not overrides.has_physical:
        return character
    document, _grip = apply_grip_selection(character.document, overrides)
    sha = hashlib.sha256(serialize_spec(document).encode("utf-8")).hexdigest()
    return replace(character, document=document, sha256=sha)


def grip_revision_summary(
    character: CompiledCharacter, overrides: GripOverrides
) -> dict[str, Any]:
    """The resolved grip, the sidecar block and the hashes of the picks (GR4).

    ``physics_spec_sha256`` is the revision's physics hash and
    ``parent_physics_spec_sha256`` the character's own: they are equal exactly
    when the picks are picture-only.  ``refit_required`` is true for a physical
    pick.
    """
    document, grip = apply_grip_selection(character.document, overrides)
    revised = compile_with_grip(character, overrides)
    visual_grip = resolve_grip(overrides, grip) if grip is not None else None
    return {
        "grip": None if grip is None else grip_spec_to_physical(grip),
        "visual": None if visual_grip is None else visual_grip_block(visual_grip),
        "spec_sha256": revised.sha256,
        "physics_spec_sha256": physics_spec_sha256(document),
        "parent_physics_spec_sha256": physics_spec_sha256(character.document),
        "spec_revision": document.get("spec_revision"),
        "refit_required": overrides.has_physical,
    }


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
    "GRIP_PRESET_CHOICES",
    "PHYSICAL_PICKS",
    "VISUAL_PICKS",
    "compile_with_grip",
    "grip_revision_summary",
    "COLOUR_CHOICES",
    "GROUND_MATERIALS",
    "appearance_library",
    "appearance_sidecar_filename",
    "build_appearance",
]
