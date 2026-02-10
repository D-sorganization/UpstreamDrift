"""Backward compatibility shim - module moved to biomechanics.swing_plane_visualization."""

import sys as _sys

from .biomechanics import swing_plane_visualization as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
