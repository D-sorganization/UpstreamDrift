"""
Humanoid Character Builder - Standalone URDF Generation Module.

A self-contained, decoupled module for generating humanoid URDF models
with video game-style character customization.

This module is designed to:
- Be completely standalone with no dependencies on other Golf Modeling Suite modules
- Provide clean, well-defined interfaces for integration
- Support parallel development without merge conflicts
- Be easily relocatable to shared tool repositories

Usage:
    from humanoid_character_builder import (
        CharacterBuilder,
        BodyParameters,
        InertiaMode,
    )

    # Create a character with custom parameters
    params = BodyParameters(
        height_m=1.80,
        mass_kg=80.0,
        build_type="athletic",
    )

    builder = CharacterBuilder()
    result = builder.build(params)

    # Export as URDF with meshes
    result.export_urdf("./output/my_humanoid")

    # Or get individual segment inertia
    inertia = builder.compute_segment_inertia(
        "left_thigh",
        mass=10.5,
        mode=InertiaMode.MESH_UNIFORM_DENSITY,
    )

Version: 0.1.0
License: Same as Golf Modeling Suite
"""

__version__ = "0.1.0"
__author__ = "Golf Modeling Suite Contributors"

# Core types - always available
from humanoid_character_builder.core.body_parameters import (
    BodyParameters,
    SegmentParameters,
    AppearanceParameters,
)
from humanoid_character_builder.core.segment_definitions import (
    SegmentDefinition,
    JointDefinition,
    HUMANOID_SEGMENTS,
    HUMANOID_JOINTS,
)
from humanoid_character_builder.core.anthropometry import (
    AnthropometryData,
    get_segment_mass_ratio,
    get_segment_length_ratio,
)

# Inertia calculation
from humanoid_character_builder.mesh.inertia_calculator import (
    InertiaMode,
    InertiaResult,
    MeshInertiaCalculator,
)
from humanoid_character_builder.mesh.primitive_inertia import (
    PrimitiveInertiaCalculator,
    PrimitiveShape,
)

# Generators
from humanoid_character_builder.generators.urdf_generator import (
    HumanoidURDFGenerator,
    URDFGeneratorConfig,
)

# Main API
from humanoid_character_builder.interfaces.api import (
    CharacterBuilder,
    CharacterBuildResult,
)

__all__ = [
    # Version
    "__version__",
    # Core types
    "BodyParameters",
    "SegmentParameters",
    "AppearanceParameters",
    "SegmentDefinition",
    "JointDefinition",
    "HUMANOID_SEGMENTS",
    "HUMANOID_JOINTS",
    # Anthropometry
    "AnthropometryData",
    "get_segment_mass_ratio",
    "get_segment_length_ratio",
    # Inertia
    "InertiaMode",
    "InertiaResult",
    "MeshInertiaCalculator",
    "PrimitiveInertiaCalculator",
    "PrimitiveShape",
    # Generators
    "HumanoidURDFGenerator",
    "URDFGeneratorConfig",
    # Main API
    "CharacterBuilder",
    "CharacterBuildResult",
]
