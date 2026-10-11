"""
Presets module for humanoid character builder.

Provides pre-configured body types and segment templates.
"""

from src.shared.python.humanoid_character_builder.presets.loader import (
    PRESET_NAMES,
    CharacterPreset,
    list_available_presets,
    list_character_presets,
    load_body_preset,
    load_character_preset,
    load_segment_template,
    validate_character_preset,
)

__all__ = [
    "CharacterPreset",
    "list_character_presets",
    "load_character_preset",
    "validate_character_preset",
    "load_body_preset",
    "load_segment_template",
    "list_available_presets",
    "PRESET_NAMES",
]
