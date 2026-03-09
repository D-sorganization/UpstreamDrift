"""
Unified Model Generation Package for URDF and Physics Simulation.

This package provides comprehensive tools for creating, editing, and
converting robot models in URDF and other formats.

Features:
- Parametric humanoid model generation
- Manual segment-by-segment construction
- Mesh-based inertia calculation (trimesh)
- URDF <-> MJCF <-> SDF format conversion
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

Lazy-loading strategy (see issue #611):
    All heavy imports are deferred to first access via ``__getattr__``.
    This breaks the circular import chain:
        __init__ -> builders.base_builder -> core.contracts -> core.validation -> core.contracts
    Only lightweight constants are imported eagerly.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"
__author__ = "Golf Modeling Suite"

# --- Only lightweight constants are imported eagerly ---
from model_generation.core.constants import (  # noqa: E402
    DEFAULT_DENSITY_KG_M3,
    DEFAULT_HEIGHT_M,
    DEFAULT_INERTIA_KG_M2,
    DEFAULT_MASS_KG,
    GRAVITY_M_S2,
)

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
    # Converters
    "URDFParser",
    "ParsedModel",
    "MJCFConverter",
    # Library
    "ModelLibrary",
    "ModelEntry",
    "ModelCategory",
    "RepositorySource",
    "ModelCache",
    # Editor
    "FrankensteinEditor",
    "URDFTextEditor",
    "ComponentType",
    "ValidationMessage",
    "ValidationSeverity",
    "DiffResult",
    # SimScape
    "SimscapeToURDFConverter",
    "MDLParser",
    "ConversionConfig",
    "convert_simscape_to_urdf",
    # REST API
    "ModelGenerationAPI",
    "APIRequest",
    "APIResponse",
    "HTTPMethod",
    # CLI
    "cli_main",
    # Convenience functions
    "quick_urdf",
    "quick_build",
]

# ---------------------------------------------------------------------------
# Lazy import mapping: name -> (module_path, attribute_name)
# See issue #611 -- all heavy imports are deferred to avoid circular
# dependency chains during package initialisation.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Core types
    "Link": ("model_generation.core.types", "Link"),
    "Joint": ("model_generation.core.types", "Joint"),
    "Inertia": ("model_generation.core.types", "Inertia"),
    "Geometry": ("model_generation.core.types", "Geometry"),
    "GeometryType": ("model_generation.core.types", "GeometryType"),
    "Material": ("model_generation.core.types", "Material"),
    "Origin": ("model_generation.core.types", "Origin"),
    "JointType": ("model_generation.core.types", "JointType"),
    "JointLimits": ("model_generation.core.types", "JointLimits"),
    "JointDynamics": ("model_generation.core.types", "JointDynamics"),
    # Validation
    "Validator": ("model_generation.core.validation", "Validator"),
    "ValidationResult": ("model_generation.core.validation", "ValidationResult"),
    "ValidationError": ("model_generation.core.validation", "ValidationError"),
    "ValidationWarning": ("model_generation.core.validation", "ValidationWarning"),
    # Inertia
    "InertiaCalculator": ("model_generation.inertia.calculator", "InertiaCalculator"),
    "InertiaMode": ("model_generation.inertia.calculator", "InertiaMode"),
    "InertiaResult": ("model_generation.inertia.calculator", "InertiaResult"),
    "box_inertia": ("model_generation.inertia.primitives", "box_inertia"),
    "cylinder_inertia": ("model_generation.inertia.primitives", "cylinder_inertia"),
    "sphere_inertia": ("model_generation.inertia.primitives", "sphere_inertia"),
    "capsule_inertia": ("model_generation.inertia.primitives", "capsule_inertia"),
    # Builders
    "BaseURDFBuilder": ("model_generation.builders.base_builder", "BaseURDFBuilder"),
    "BuildResult": ("model_generation.builders.base_builder", "BuildResult"),
    "ManualBuilder": ("model_generation.builders.manual_builder", "ManualBuilder"),
    "Handedness": ("model_generation.builders.manual_builder", "Handedness"),
    "ParametricBuilder": (
        "model_generation.builders.parametric_builder",
        "ParametricBuilder",
    ),
    "ParametricConfig": (
        "model_generation.builders.parametric_builder",
        "ParametricConfig",
    ),
    "URDFWriter": ("model_generation.builders.urdf_writer", "URDFWriter"),
    # Converters
    "URDFParser": ("model_generation.converters.urdf_parser", "URDFParser"),
    "ParsedModel": ("model_generation.converters.urdf_parser", "ParsedModel"),
    "MJCFConverter": ("model_generation.converters.mjcf_converter", "MJCFConverter"),
    # SimScape
    "SimscapeToURDFConverter": (
        "model_generation.converters.simscape",
        "SimscapeToURDFConverter",
    ),
    "MDLParser": ("model_generation.converters.simscape", "MDLParser"),
    "ConversionConfig": (
        "model_generation.converters.simscape",
        "ConversionConfig",
    ),
    "convert_simscape_to_urdf": (
        "model_generation.converters.simscape",
        "convert_simscape_to_urdf",
    ),
    # Library
    "ModelLibrary": ("model_generation.library", "ModelLibrary"),
    "ModelEntry": ("model_generation.library", "ModelEntry"),
    "ModelCategory": ("model_generation.library", "ModelCategory"),
    "RepositorySource": ("model_generation.library", "RepositorySource"),
    "ModelCache": ("model_generation.library", "ModelCache"),
    # Editor
    "FrankensteinEditor": ("model_generation.editor", "FrankensteinEditor"),
    "URDFTextEditor": ("model_generation.editor", "URDFTextEditor"),
    "ComponentType": ("model_generation.editor", "ComponentType"),
    "ValidationMessage": ("model_generation.editor", "ValidationMessage"),
    "ValidationSeverity": ("model_generation.editor", "ValidationSeverity"),
    "DiffResult": ("model_generation.editor", "DiffResult"),
    # REST API
    "ModelGenerationAPI": ("model_generation.api", "ModelGenerationAPI"),
    "APIRequest": ("model_generation.api", "APIRequest"),
    "APIResponse": ("model_generation.api", "APIResponse"),
    "HTTPMethod": ("model_generation.api", "HTTPMethod"),
    # CLI
    "cli_main": ("model_generation.cli", "main"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load attributes on first access (see issue #611)."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache on the module so subsequent accesses are fast
        globals()[name] = value
        return value

    # Convenience functions are resolved here (they need lazy deps)
    if name == "quick_urdf":
        globals()["quick_urdf"] = quick_urdf
        return quick_urdf
    if name == "quick_build":
        globals()["quick_build"] = quick_build
        return quick_build

    raise AttributeError(f"module 'model_generation' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Shared preset configuration (DRY: single source of truth used by both
# quick_urdf() and quick_build() -- eliminates the duplicated local dict).
# ---------------------------------------------------------------------------

_HUMANOID_PRESETS: dict[str, dict[str, float]] = {
    "athletic": {"gender_factor": 0.7, "shoulder_width_factor": 1.1},
    "average": {"gender_factor": 0.5},
    "heavy": {"gender_factor": 0.5, "hip_width_factor": 1.15},
    "lean": {"gender_factor": 0.5, "shoulder_width_factor": 0.95},
}


# ---------------------------------------------------------------------------
# Convenience functions (use lazy imports internally)
# ---------------------------------------------------------------------------


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
    from model_generation.builders.parametric_builder import ParametricBuilder

    builder = ParametricBuilder(robot_name)

    if preset:
        preset_config = _HUMANOID_PRESETS.get(preset.lower(), {})
        builder.set_parameters(height_m=height_m, mass_kg=mass_kg, **preset_config)
    else:
        builder.set_parameters(height_m=height_m, mass_kg=mass_kg)

    builder.add_humanoid_segments()
    result = builder.build()

    if not result.success:
        raise ValueError(f"Failed to generate URDF: {result.error_message}")

    return str(result.urdf_xml)


def quick_build(
    height_m: float = DEFAULT_HEIGHT_M,
    mass_kg: float = DEFAULT_MASS_KG,
    preset: str | None = None,
    output_path: str | None = None,
) -> Any:
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
    from pathlib import Path

    from model_generation.builders.parametric_builder import ParametricBuilder

    builder = ParametricBuilder("humanoid")

    if preset:
        preset_config = _HUMANOID_PRESETS.get(preset.lower(), {})
        builder.set_parameters(height_m=height_m, mass_kg=mass_kg, **preset_config)
    else:
        builder.set_parameters(height_m=height_m, mass_kg=mass_kg)

    builder.add_humanoid_segments()
    result = builder.build()

    if output_path and result.success and result.urdf_xml is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.urdf_xml)
        result.output_path = path

    return result
