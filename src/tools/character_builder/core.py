"""Headless core of the Character Builder tool (CMB-3, #11654).

Holds the editable parameters, applies presets and compiles/exports through
the shared ``humanoid_character_builder`` modules; no Qt, so it is unit
tested in any environment (AGENTS.md section E).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.api.services.character_appearance_service import (
    appearance_library as _appearance_library,
)
from src.api.services.character_appearance_service import (
    appearance_sidecar_filename,
    build_appearance,
)
from src.shared.python.humanoid_character_builder import spec_export
from src.shared.python.humanoid_character_builder.presets.loader import (
    CharacterPreset,
    list_character_presets,
    load_character_preset,
)
from src.shared.python.humanoid_character_builder.spec_params import (
    PARAMETER_RANGES,
    SpecCharacterParameters,
)

__all__ = ["APPEARANCE_NAMES", "CharacterBuilderModel", "PARAMETER_RANGES"]

# Appearance choices accepted by set_appearance(); matches build_appearance's
# optional fields (src/api/services/character_appearance_service.py) minus
# "name", which this tool does not let the user set.
APPEARANCE_NAMES = (
    "skin_tone",
    "clothing",
    "club_finish",
    "headwear",
    "headwear_material",
    "ground_material",
)


class CharacterBuilderModel:
    """Editable character: parameters, optional preset, compile and export."""

    def __init__(self, preset_id: str | None = None) -> None:
        self._preset: CharacterPreset | None = None
        self._params = SpecCharacterParameters()
        self._appearance_choices: dict[str, Any] = {}
        if preset_id is not None:
            self.apply_preset(preset_id)

    @property
    def parameters(self) -> SpecCharacterParameters:
        """Current parameters (immutable value object)."""
        return self._params

    def parameters_dict(self) -> dict[str, Any]:
        """Current parameters as a plain mapping (delegates for callers)."""
        return self._params.to_dict()

    @property
    def preset(self) -> CharacterPreset | None:
        """The preset last applied, or ``None`` once a value is edited."""
        return self._preset

    @staticmethod
    def preset_ids() -> list[str]:
        """Shipped preset ids, sorted."""
        return list_character_presets()

    def apply_preset(self, preset_id: str) -> None:
        """Replace all parameters with the preset's (``ValueError`` if unknown)."""
        preset = load_character_preset(preset_id)
        self._preset, self._params = preset, preset.parameters

    def set_parameter(self, name: str, value: float | str) -> None:
        """Change one parameter, validating the full set before committing."""
        if name not in self._params.to_dict():
            raise ValueError(f"unknown character parameter: {name!r}")
        overrides: dict[str, Any] = {name: value}
        self._params = replace(self._params, **overrides)
        self._preset = None

    def compile(self) -> spec_export.CompiledCharacter:
        """Compile the current parameters to a full-body spec."""
        return spec_export.compile_character(None, self._params.to_dict())

    def summary_text(self) -> str:
        """Plain-text summary of the compiled character for display."""
        built = self.compile()
        s = spec_export.build_summary(built)
        lines = [
            f"Preset: {self._preset.name if self._preset else 'Custom'}",
            f"Stature: {s['parameters']['stature_m']:.2f} m   "
            f"Mass setting: {s['parameters']['mass_kg']:.1f} kg   Club: {s['club']}",
            f"Bodies: {s['bodies']}   Joints: {s['joints']}   "
            f"Coordinates: {s['coordinates']}",
            f"Total model mass (with club and hands): {s['total_mass_kg']:.1f} kg",
            f"Spec SHA-256: {s['spec_sha256']}",
            f"Qualification: {s['qualification']}",
        ]
        if self._preset is not None:
            lines.append(f"Limitations: {self._preset.limitations}")
        return "\n".join(lines)

    def export(self, fmt: str, directory: Path | str) -> Path:
        """Write ``spec``/``urdf``/``mjcf``/``osim``/``appearance`` into ``directory``."""
        if fmt == "appearance":
            return self.export_appearance(directory)
        text, _media, filename = spec_export.export_character(self.compile(), fmt)
        return self._write_export(directory, filename, text)

    def as_dict(self) -> dict[str, Any]:
        """Parameters as a plain mapping (for save/restore)."""
        return self._params.to_dict()

    def set_appearance(self, name: str, value: str) -> None:
        """Set one appearance choice, validating the merged document first.

        Precondition: ``name`` must be one of :data:`APPEARANCE_NAMES`.
        Raises ``ValueError`` for an unknown name, or an unknown skin tone,
        clothing preset, club finish, headwear or ground material value;
        the current choices are left unchanged when it raises.
        Postcondition: ``appearance_document()`` validates against the
        appearance-v1 schema.
        """
        if name not in APPEARANCE_NAMES:
            raise ValueError(f"unknown appearance choice: {name!r}")
        candidate = {**self._appearance_choices, name: value}
        build_appearance(candidate, self.compile().sha256)
        self._appearance_choices = candidate

    def appearance_document(self) -> dict[str, Any]:
        """Appearance document for the current choices, bound to the spec hash."""
        return build_appearance(self._appearance_choices, self.compile().sha256)

    def appearance_library(self) -> dict[str, Any]:
        """Pickable appearance names and materials (delegates for the GUI)."""
        return _appearance_library()

    def export_appearance(self, directory: Path | str) -> Path:
        """Write the appearance-v1 sidecar bound to the compiled spec hash.

        Postcondition: the filename stem equals the stem of this model's
        spec export (``spec_export.export_character``), so the sidecar sits
        beside its spec as ``<stem>.appearance.json``.
        """
        compiled = self.compile()
        doc = build_appearance(self._appearance_choices, compiled.sha256)
        filename = appearance_sidecar_filename(
            doc, preset_id=compiled.preset_id or "custom", spec_sha256=compiled.sha256
        )
        text = json.dumps(doc, indent=2) + "\n"
        return self._write_export(directory, filename, text)

    @staticmethod
    def _write_export(directory: Path | str, filename: str, text: str) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        path.write_text(text, encoding="utf-8")
        return path
