"""Backward compatibility shim - module moved to validation_pkg.validation."""

import sys as _sys

from .validation_pkg import validation as _real_module  # noqa: E402
from .validation_pkg.validation import (  # noqa: F401
    F,
    PhysicalValidationError,
    logger,
    validate_friction_coefficient,
    validate_inertia_matrix,
    validate_joint_limits,
    validate_mass,
    validate_physical_bounds,
    validate_timestep,
)

_sys.modules[__name__] = _real_module
