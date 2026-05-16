"""
Humanoid model generation components.

This module re-exports components from the humanoid_character_builder
for integration with the unified model_generation package.
"""

from __future__ import annotations

from humanoid_character_builder import (
    AnthropometryData,
    BodyParameters,
    CharacterBuilder,
    CharacterBuildResult,
    HumanoidURDFGenerator,
    InertiaMode,
    InertiaResult,
    MeshInertiaCalculator,
    PrimitiveInertiaCalculator,
    PrimitiveShape,
    URDFGeneratorConfig,
    get_segment_length_ratio,
    get_segment_mass_ratio,
)

from humanoid_character_builder.core.anthropometry import (
    DE_LEVA_DATA,
    estimate_segment_dimensions,
    estimate_segment_inertia_from_gyration,
    estimate_segment_masses,
    get_com_location,
)
from humanoid_character_builder.core.body_parameters import (
    AppearanceParameters,
    BuildType,
    GenderModel,
    SegmentParameters,
)
from humanoid_character_builder.core.segment_definitions import (
    HUMANOID_JOINTS,
    HUMANOID_SEGMENTS,
    JointDefinition,
    SegmentDefinition,
)
from humanoid_character_builder.interfaces import (
    ExportOptions,
    SegmentMeshInfo,
    quick_build,
    quick_urdf,
)
from humanoid_character_builder.presets.loader import (
    PRESET_NAMES,
    get_preset_info,
    list_available_presets,
    load_body_preset,
)

__all__: list[str] = [
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
    "AnthropometryData",
    # Segments
    "HUMANOID_SEGMENTS",
    "HUMANOID_JOINTS",
    "SegmentDefinition",
    "JointDefinition",
    # Presets
    "PRESET_NAMES",
    "load_body_preset",
    "list_available_presets",
    "get_preset_info",
    # Generators / API
    "CharacterBuilder",
    "CharacterBuildResult",
    "ExportOptions",
    "SegmentMeshInfo",
    "HumanoidURDFGenerator",
    "URDFGeneratorConfig",
    "quick_build",
    "quick_urdf",
    # Inertia
    "InertiaMode",
    "InertiaResult",
    "MeshInertiaCalculator",
    "PrimitiveInertiaCalculator",
    "PrimitiveShape",
]
