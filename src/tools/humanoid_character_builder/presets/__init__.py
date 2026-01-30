"""
Presets module for humanoid character builder.

Provides pre-configured body types and segment templates.
"""

from humanoid_character_builder.presets.loader import (
    load_body_preset,
    load_segment_template,
    list_available_presets,
    PRESET_NAMES,
)

__all__ = [
    "load_body_preset",
    "load_segment_template",
    "list_available_presets",
    "PRESET_NAMES",
]
