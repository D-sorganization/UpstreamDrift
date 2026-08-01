# mypy: ignore-errors
"""Humanoid model generation components.

Re-exports from humanoid_character_builder for integration with the
unified model_generation package.
"""

try:
    from shared.python.humanoid_character_builder import (
        DE_LEVA_DATA,
        PRESET_NAMES,
        AnthropometryData,
        SegmentMeshInfo,
    )
    from shared.python.humanoid_character_builder.appearance import AppearanceParameters
    from shared.python.humanoid_character_builder.builder import (
        CharacterBuilder,
        CharacterBuildResult,
        ExportOptions,
        quick_build,
        quick_urdf,
    )
    from shared.python.humanoid_character_builder.core.body_parameters import (
        BodyParameters,
        BuildType,
        GenderModel,
        SegmentParameters,
    )
    from shared.python.humanoid_character_builder.mesh import (
        InertiaMode,
        InertiaResult,
        MeshInertiaCalculator,
        PrimitiveInertiaCalculator,
        PrimitiveShape,
    )
    from shared.python.humanoid_character_builder.presets.loader import (
        get_preset_info,
        list_available_presets,
        load_body_preset,
    )
    from shared.python.humanoid_character_builder.segments import (
        HUMANOID_JOINTS,
        HUMANOID_SEGMENTS,
        JointDefinition,
        SegmentDefinition,
    )
    from shared.python.humanoid_character_builder.urdf import (
        HumanoidURDFGenerator,
        URDFGeneratorConfig,
    )
except ImportError:  # pragma: no cover
    pass

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
