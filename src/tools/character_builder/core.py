"""Headless core of the Character Builder tool (CMB-3, #11654).

Holds the editable parameters, applies presets and compiles/exports through
the shared ``humanoid_character_builder`` modules; no Qt, so it is unit
tested in any environment (AGENTS.md section E).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

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

__all__ = ["CharacterBuilderModel", "PARAMETER_RANGES"]


class CharacterBuilderModel:
    """Editable character: parameters, optional preset, compile and export."""

    def __init__(self, preset_id: str | None = None) -> None:
        self._preset: CharacterPreset | None = None
        self._params = SpecCharacterParameters()
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
        self._params = replace(self._params, **{name: value})
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
        """Write ``spec``/``urdf``/``mjcf``/``osim`` into ``directory``."""
        text, _media, filename = spec_export.export_character(self.compile(), fmt)
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        path.write_text(text, encoding="utf-8")
        return path

    def as_dict(self) -> dict[str, Any]:
        """Parameters as a plain mapping (for save/restore)."""
        return self._params.to_dict()
