"""Backward compatibility shim - module moved to biomechanics.muscle_equilibrium."""

import sys as _sys

from .biomechanics import muscle_equilibrium as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
