"""Humanoid model generation components.

Re-exports from humanoid_character_builder for integration with the
unified model_generation package.
"""

from __future__ import annotations

from src.shared.python.humanoid_character_builder.core.anthropometry import (
    AnthropometryData,
    DE_LEVA_DATA,
    estimate_segment_dimensions,
    estimate_segment_inertia_from_gyration,
    estimate_segment_masses,
    get_com_location,
    get_segment_length_ratio,
    get_segment_mass_ratio,
)
from src.shared.python.humanoid_character_builder.core.body_parameters import (
    AppearanceParameters,
    BodyParameters,
    BuildType,
    GenderModel,
    SegmentParameters,
)
from src.shared.python.humanoid_character_builder.core.segment_definitions import (
    HUMANOID_JOINTS,
    HUMANOID_SEGMENTS,
    JointDefinition,
    SegmentDefinition,
)
from src.shared.python.humanoid_character_builder.generators.urdf_config import (
    URDFGeneratorConfig,
)
from src.shared.python.humanoid_character_builder.generators.urdf_generator import (
    HumanoidURDFGenerator,
)
from src.shared.python.humanoid_character_builder.interfaces.api import (
    CharacterBuildResult,
    CharacterBuilder,
    ExportOptions,
    SegmentMeshInfo,
    quick_build,
    quick_urdf,
)
from src.shared.python.humanoid_character_builder.mesh.inertia_calculator import (
    InertiaMode,
    InertiaResult,
    MeshInertiaCalculator,
)
from src.shared.python.humanoid_character_builder.mesh.primitive_inertia import (
    PrimitiveInertiaCalculator,
    PrimitiveShape,
)
from src.shared.python.humanoid_character_builder.presets.loader import (
    PRESET_NAMES,
    get_preset_info,
    list_available_presets,
    load_body_preset,
)

__all__: list[str] = [
    # Body parameters & appearance
    "AppearanceParameters",
    "BodyParameters",
    "BuildType",
    "GenderModel",
    "SegmentParameters",
    # Anthropometry
    "AnthropometryData",
    "DE_LEVA_DATA",
    "estimate_segment_masses",
    "estimate_segment_dimensions",
    "estimate_segment_inertia_from_gyration",
    "get_segment_mass_ratio",
    "get_segment_length_ratio",
    "get_com_location",
    # Segments & joints
    "HUMANOID_SEGMENTS",
    "HUMANOID_JOINTS",
    "SegmentDefinition",
    "JointDefinition",
    # Presets
    "PRESET_NAMES",
    "load_body_preset",
    "list_available_presets",
    "get_preset_info",
    # Builder
    "CharacterBuilder",
    "CharacterBuildResult",
    "ExportOptions",
    "SegmentMeshInfo",
    # URDF
    "HumanoidURDFGenerator",
    "URDFGeneratorConfig",
    # Convenience
    "quick_build",
    "quick_urdf",
    # Mesh/inertia
    "InertiaMode",
    "InertiaResult",
    "MeshInertiaCalculator",
    "PrimitiveInertiaCalculator",
    "PrimitiveShape",
]
