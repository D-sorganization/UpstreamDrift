"""
Humanoid model generation components.

This module re-exports components from the humanoid_character_builder
for integration with the unified model_generation package.
"""

from __future__ import annotations

# Import from humanoid_character_builder
try:
    from humanoid_character_builder.core.body_parameters import (
        BodyParameters,
        BuildType,
        GenderModel,
    )
    from humanoid_character_builder.presets.loader import (
        load_body_preset,
    )

    __all__ = [
        # Body parameters
        "BodyParameters",
        "BuildType",
        "GenderModel",
        "AppearanceParameters",
        "SegmentParameters",
        # Anthropometry
        "DE_LEVA_DATA",
        "estimate_segment_masses",
        "estimate_segment_dimensions",
        "estimate_segment_inertia_from_gyration",
        "get_segment_mass_ratio",
        "get_segment_length_ratio",
        "get_com_location",
        # Segments
        "HUMANOID_SEGMENTS",
        "SegmentDefinition",
        # Presets
        "PRESET_NAMES",
        "load_body_preset",
        "list_available_presets",
        "get_preset_info",
    ]

except ImportError:
    # humanoid_character_builder not available
    __all__ = []

    def _not_available(*args, **kwargs):
        raise ImportError(
            "humanoid_character_builder not available. Ensure it is in the Python path."
        )

    # Create stubs
    BodyParameters = _not_available
    BuildType = _not_available
    GenderModel = _not_available
    load_body_preset = _not_available
