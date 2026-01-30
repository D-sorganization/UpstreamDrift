"""
Unified Model Generation Package for URDF and Physics Simulation.

This package provides comprehensive tools for creating, editing, and
converting robot models in URDF and other formats.

Features:
- Parametric humanoid model generation
- Manual segment-by-segment construction
- Mesh-based inertia calculation (trimesh)
- URDF ↔ MJCF ↔ SDF format conversion
- Model library with repository integration
- Frankenstein editor for component composition
- Text-based URDF editing with diff view

Quick Start:
    # Generate a humanoid URDF
    from model_generation import quick_urdf
    urdf = quick_urdf(height_m=1.80, preset="athletic")

    # Full parametric build
    from model_generation import ModelBuilder
    builder = ModelBuilder()
    result = builder.build_humanoid(height_m=1.85, mass_kg=85.0)
    result.save("my_humanoid.urdf")

    # Manual construction
    from model_generation import ManualBuilder, Link, Joint, Inertia
    builder = ManualBuilder("robot")
    builder.add_link(Link(name="base", inertia=Inertia.from_box(10, 1, 1, 0.5)))
    urdf = builder.build().urdf_xml

    # Load from library
    from model_generation import ModelLibrary
    library = ModelLibrary()
    model = library.load("human_gazebo/adult_male")

    # Convert formats
    from model_generation import convert_urdf_to_mjcf
    mjcf = convert_urdf_to_mjcf("robot.urdf")
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Golf Modeling Suite"

# Core types
# Builders
from model_generation.builders.base_builder import BaseURDFBuilder, BuildResult
from model_generation.builders.manual_builder import Handedness, ManualBuilder
from model_generation.builders.parametric_builder import (
    ParametricBuilder,
    ParametricConfig,
)
from model_generation.builders.urdf_writer import URDFWriter

# Constants
from model_generation.core.constants import (
    DEFAULT_DENSITY_KG_M3,
    DEFAULT_HEIGHT_M,
    DEFAULT_INERTIA_KG_M2,
    DEFAULT_MASS_KG,
    GRAVITY_M_S2,
)
from model_generation.core.types import (
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Material,
    Origin,
)

# Validation
from model_generation.core.validation import (
    ValidationError,
    ValidationResult,
    ValidationWarning,
    Validator,
)

# Inertia calculation
from model_generation.inertia.calculator import (
    InertiaCalculator,
    InertiaMode,
    InertiaResult,
)
from model_generation.inertia.primitives import (
    box_inertia,
    capsule_inertia,
    cylinder_inertia,
    sphere_inertia,
)

# Public API will be imported when available
# from model_generation.api.builder_api import ModelBuilder
# from model_generation.api.quick import quick_build, quick_urdf

__all__ = [
    # Version
    "__version__",
    # Core types
    "Link",
    "Joint",
    "Inertia",
    "Geometry",
    "GeometryType",
    "Material",
    "Origin",
    "JointType",
    "JointLimits",
    "JointDynamics",
    # Validation
    "Validator",
    "ValidationResult",
    "ValidationError",
    "ValidationWarning",
    # Constants
    "GRAVITY_M_S2",
    "DEFAULT_DENSITY_KG_M3",
    "DEFAULT_INERTIA_KG_M2",
    "DEFAULT_HEIGHT_M",
    "DEFAULT_MASS_KG",
    # Inertia
    "InertiaCalculator",
    "InertiaMode",
    "InertiaResult",
    "box_inertia",
    "cylinder_inertia",
    "sphere_inertia",
    "capsule_inertia",
    # Builders
    "BaseURDFBuilder",
    "BuildResult",
    "ManualBuilder",
    "Handedness",
    "ParametricBuilder",
    "ParametricConfig",
    "URDFWriter",
]


# Convenience functions
def quick_urdf(
    height_m: float = DEFAULT_HEIGHT_M,
    mass_kg: float = DEFAULT_MASS_KG,
    preset: str | None = None,
    robot_name: str = "humanoid",
) -> str:
    """
    Generate a humanoid URDF quickly with minimal configuration.

    Args:
        height_m: Height in meters
        mass_kg: Mass in kg
        preset: Optional preset name (athletic, average, heavy, lean)
        robot_name: Name for the robot element

    Returns:
        URDF XML string

    Example:
        urdf = quick_urdf(height_m=1.85, preset="athletic")
    """
    builder = ParametricBuilder(robot_name)

    # Apply preset if specified
    if preset:
        presets = {
            "athletic": {"gender_factor": 0.7, "shoulder_width_factor": 1.1},
            "average": {"gender_factor": 0.5},
            "heavy": {"gender_factor": 0.5, "hip_width_factor": 1.15},
            "lean": {"gender_factor": 0.5, "shoulder_width_factor": 0.95},
        }
        preset_config = presets.get(preset.lower(), {})
        builder.set_parameters(height_m=height_m, mass_kg=mass_kg, **preset_config)
    else:
        builder.set_parameters(height_m=height_m, mass_kg=mass_kg)

    builder.add_humanoid_segments()
    result = builder.build()

    if not result.success:
        raise ValueError(f"Failed to generate URDF: {result.error_message}")

    return result.urdf_xml


def quick_build(
    height_m: float = DEFAULT_HEIGHT_M,
    mass_kg: float = DEFAULT_MASS_KG,
    preset: str | None = None,
    output_path: str | None = None,
) -> BuildResult:
    """
    Build a humanoid model quickly with minimal configuration.

    Args:
        height_m: Height in meters
        mass_kg: Mass in kg
        preset: Optional preset name
        output_path: Optional path to save URDF

    Returns:
        BuildResult with URDF and metadata

    Example:
        result = quick_build(height_m=1.80, output_path="./humanoid.urdf")
    """
    builder = ParametricBuilder("humanoid")

    if preset:
        presets = {
            "athletic": {"gender_factor": 0.7, "shoulder_width_factor": 1.1},
            "average": {"gender_factor": 0.5},
            "heavy": {"gender_factor": 0.5, "hip_width_factor": 1.15},
            "lean": {"gender_factor": 0.5, "shoulder_width_factor": 0.95},
        }
        preset_config = presets.get(preset.lower(), {})
        builder.set_parameters(height_m=height_m, mass_kg=mass_kg, **preset_config)
    else:
        builder.set_parameters(height_m=height_m, mass_kg=mass_kg)

    builder.add_humanoid_segments()
    result = builder.build()

    if output_path and result.success:
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.urdf_xml)
        result.output_path = path

    return result
