"""Backward compatibility shim - module moved to biomechanics.swing_comparison."""
import sys as _sys

from .biomechanics import swing_comparison as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
