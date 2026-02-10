"""Backward compatibility shim - module moved to spatial_algebra.manipulability."""

import sys as _sys

from .spatial_algebra import manipulability as _real_module  # noqa: E402
from .spatial_algebra.manipulability import (  # noqa: F401
    CATASTROPHIC_SINGULARITY_THRESHOLD,
    SINGULARITY_FALLBACK_THRESHOLD,
    SINGULARITY_WARNING_THRESHOLD,
    SingularityError,
    check_jacobian_conditioning,
    compute_manipulability_ellipsoid,
    compute_manipulability_index,
    get_jacobian_conditioning,
    logger,
)

_sys.modules[__name__] = _real_module
