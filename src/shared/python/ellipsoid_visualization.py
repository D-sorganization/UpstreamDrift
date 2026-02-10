"""Backward compatibility shim - module moved to gui_pkg.ellipsoid_visualization."""

import sys as _sys

from .gui_pkg import ellipsoid_visualization as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
