"""Backward compatibility shim - module moved to validation_pkg.validation_helpers."""

import sys as _sys

from .validation_pkg import validation_helpers as _real_module  # noqa: E402
from .validation_pkg.validation_helpers import (  # noqa: F401
    MAX_CARTESIAN_ACCELERATION_M_S2,
    MAX_CARTESIAN_VELOCITY_M_S,
    MAX_JOINT_ACCELERATION_RAD_S2,
    MAX_JOINT_POSITION_RAD,
    MAX_JOINT_VELOCITY_RAD_S,
    PhysicsValidationError,
    ValidationLevel,
    validate_cartesian_state,
    validate_finite,
    validate_joint_state,
    validate_magnitude,
    validate_model_parameters,
)

_sys.modules[__name__] = _real_module
