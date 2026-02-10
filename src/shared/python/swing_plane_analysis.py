"""Backward compatibility shim - module moved to biomechanics.swing_plane_analysis."""
import sys as _sys

from .biomechanics import swing_plane_analysis as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
