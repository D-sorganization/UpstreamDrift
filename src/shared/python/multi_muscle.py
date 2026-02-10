"""Backward compatibility shim - module moved to biomechanics.multi_muscle."""

import sys as _sys

from .biomechanics import multi_muscle as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
