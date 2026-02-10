"""Backward compatibility shim - module moved to spatial_algebra.indexed_acceleration."""

import sys as _sys

from .spatial_algebra import indexed_acceleration as _real_module  # noqa: E402
from .spatial_algebra.indexed_acceleration import (  # noqa: F401
    AccelerationClosureError,
    IndexedAcceleration,
    compute_indexed_acceleration_from_engine,
)

_sys.modules[__name__] = _real_module
