"""Backward compatibility shim - module moved to core.numerical_constants."""

import sys as _sys

from .core import numerical_constants as _real_module  # noqa: E402
from .core.numerical_constants import (  # noqa: F401
    CONDITION_NUMBER_CRITICAL_THRESHOLD,
    CONDITION_NUMBER_WARNING_THRESHOLD,
    EPSILON_FINITE_DIFF_JACOBIAN,
    EPSILON_MASS_MATRIX_REGULARIZATION,
    EPSILON_SINGULARITY_DETECTION,
    GRAVITY_STANDARD,
    HUMAN_BODY_MASS_PLAUSIBLE_RANGE,
    SEGMENT_LENGTH_TO_HEIGHT_RATIO_PLAUSIBLE,
    TOLERANCE_ENERGY_CONSERVATION,
    TOLERANCE_WORK_ENERGY_MISMATCH,
)

_sys.modules[__name__] = _real_module
