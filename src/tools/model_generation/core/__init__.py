"""
Core types and utilities for model generation.

This module provides the foundational data structures and validation
logic used throughout the model_generation package.
"""

from model_generation.core.constants import (
    DEFAULT_DENSITY_KG_M3,
    DEFAULT_INERTIA_KG_M2,
    DEFAULT_JOINT_DAMPING,
    DEFAULT_JOINT_EFFORT,
    DEFAULT_JOINT_FRICTION,
    DEFAULT_JOINT_VELOCITY,
    GRAVITY_M_S2,
    INTERMEDIATE_LINK_MASS,
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
from model_generation.core.validation import (
    ValidationError,
    ValidationResult,
    ValidationWarning,
    Validator,
)

__all__ = [
    # Types
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
    "DEFAULT_JOINT_DAMPING",
    "DEFAULT_JOINT_FRICTION",
    "DEFAULT_JOINT_EFFORT",
    "DEFAULT_JOINT_VELOCITY",
    "INTERMEDIATE_LINK_MASS",
]
