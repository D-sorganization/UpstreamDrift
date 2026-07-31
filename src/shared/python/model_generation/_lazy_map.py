"""Lazy import dispatch table for ``model_generation``.

Extracted from ``__init__.py`` (issue #1696) to keep the package entry-point
below 120 lines.  Each entry maps a public name to the (module_path,
attribute_name) pair that provides it.
"""

from __future__ import annotations

LAZY_IMPORTS: dict[str, tuple[str, str]] = {
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
