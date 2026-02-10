"""Backward compatibility shim - module moved to biomechanics.activation_dynamics."""
import sys as _sys

from .biomechanics import activation_dynamics as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
