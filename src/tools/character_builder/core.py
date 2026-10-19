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
    compile_with_grip,
)
from src.shared.python.motion_matching.execution.grip_selection import (
    PHYSICAL_PICKS,
    VISUAL_PICKS,
    GripOverrides,
    apply_grip_selection,
)
from src.shared.python.motion_matching.grip_contract import GripSpec
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
    "skin_color",
    "hair_color",
    "top_color",
    "bottom_color",
    "glove",
)


class CharacterBuilderModel:
    """Editable character: parameters, optional preset, compile and export."""

    def __init__(self, preset_id: str | None = None) -> None:
        self._preset: CharacterPreset | None = None
        self._params = SpecCharacterParameters()
        self._appearance_choices: dict[str, Any] = {}
        self._grip_picks: dict[str, Any] = {}
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
        built = spec_export.compile_character(None, self._params.to_dict())
        return compile_with_grip(built, self.grip_overrides())

    def set_grip(self, name: str, value: float | str | None) -> None:
        """Set (or, with ``None``, clear) one grip pick (HRG-GR4, #13205).

        Physical picks (``preset``, ``lead_shaft_rotation``, ..., ``grip_style``)
        change the compiled spec into a new revision that needs a refit;
        ``grip_thumb_offset_deg`` is picture-only and goes in the appearance
        sidecar.  ``ValueError`` for an unknown pick or an invalid value; the
        picks are unchanged then.
        """
        if name not in (*PHYSICAL_PICKS, *VISUAL_PICKS):
            raise ValueError(f"unknown grip pick: {name!r}")
        candidate = {
            k: v for k, v in {**self._grip_picks, name: value}.items() if v is not None
        }
        picks = GripOverrides.from_picks(candidate)  # validates the pick values
        built = spec_export.compile_character(None, self._params.to_dict())
        compile_with_grip(built, picks)  # and the resolved grip's ranges
        self._grip_picks = candidate

    def resolved_grip(self) -> GripSpec | None:
        """The grip the picks resolve to (``None``: picture-only, fit not a grip)."""
        built = spec_export.compile_character(None, self._params.to_dict())
        return apply_grip_selection(built.document, self.grip_overrides())[1]

    def grip_overrides(self) -> GripOverrides:
        """The grip picks made so far."""
        return GripOverrides.from_picks(self._grip_picks)

    def grip_picks(self) -> dict[str, Any]:
        """The explicit grip picks made so far (a copy)."""
        return dict(self._grip_picks)

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
        lines.extend(self._grip_lines())
        if self._preset is not None:
            lines.append(f"Limitations: {self._preset.limitations}")
        return "\n".join(lines)

    def _grip_lines(self) -> list[str]:
        """Summary lines of the grip picks: what changed and that it needs a refit."""
        grip = self.resolved_grip()
        if grip is None or not self._grip_picks:
            return []
        lines = [
            f"Grip: lead {grip.lead.alpha_deg:.1f} deg, trail "
            f"{grip.trail.alpha_deg:.1f} deg about the shaft (positive is stronger); "
            f"{grip.style.value}"
        ]
        base = spec_export.compile_character(None, self._params.to_dict())
        if self.compile().sha256 != base.sha256:
            lines.append("Grip change makes a new model revision: refit it.")
        return lines

    def export(self, fmt: str, directory: Path | str) -> Path:
        """Write ``spec``/``urdf``/``mjcf``/``osim``/``appearance`` into ``directory``."""
        if fmt == "appearance":
            return self.export_appearance(directory)
        text, _media, filename = spec_export.export_character(self.compile(), fmt)
        return self._write_export(directory, filename, text)

    def verify_cross_engine_parity(
        self, *, mass_tol_kg: float = 1e-9, frame_tol_m: float = 1e-3
    ) -> dict[str, Any]:
        """Verify cross-engine L0 parity for the compiled model (CMB-11, #11662)."""
        return spec_export.verify_cross_engine_l0_parity(
            self.compile(), mass_tol_kg=mass_tol_kg, frame_tol_m=frame_tol_m
        )

    def as_dict(self) -> dict[str, Any]:
        """Parameters as a plain mapping (for save/restore)."""
        return self._params.to_dict()

    def set_appearance(self, name: str, value: str | None) -> None:
        """Set one appearance choice, validating the merged document first.

        Precondition: ``name`` must be one of :data:`APPEARANCE_NAMES`. The
        ``*_color`` names take ``#rrggbb`` and are written as document
        ``materials`` entries (HRG-B2); ``None`` clears a choice.
        Raises ``ValueError`` for an unknown name, or an unknown skin tone,
        clothing preset, club finish, headwear or ground material value;
        the current choices are left unchanged when it raises.
        Postcondition: ``appearance_document()`` validates against the
        appearance-v1 schema.
        """
        if name not in APPEARANCE_NAMES:
            raise ValueError(f"unknown appearance choice: {name!r}")
        candidate = {**self._appearance_choices, name: value}
        if value is None:
            candidate.pop(name)
        build_appearance(candidate, self.compile().sha256)
        self._appearance_choices = candidate

    def appearance_document(self) -> dict[str, Any]:
        """Appearance document for the current choices, bound to the spec hash."""
        return build_appearance(self._sidecar_choices(), self.compile().sha256)

    def _sidecar_choices(self) -> dict[str, Any]:
        """Appearance choices plus the picture-only grip picks (never physical)."""
        visual = self.grip_overrides().visual
        return {**self._appearance_choices, **visual}

    def appearance_choices(self) -> dict[str, Any]:
        """The explicit appearance choices made so far (a copy)."""
        return dict(self._appearance_choices)

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
        doc = build_appearance(self._sidecar_choices(), compiled.sha256)
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
