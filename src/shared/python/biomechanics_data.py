"""Backward compatibility shim - module moved to biomechanics.biomechanics_data."""

import sys as _sys

from .biomechanics import biomechanics_data as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
