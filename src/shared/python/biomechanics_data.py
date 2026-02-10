"""Backward compatibility shim - module moved to biomechanics.biomechanics_data."""

import sys as _sys

from .biomechanics import biomechanics_data as _real_module  # noqa: E402
from .biomechanics.biomechanics_data import (  # noqa: F401
    BiomechanicalData,
)

_sys.modules[__name__] = _real_module
