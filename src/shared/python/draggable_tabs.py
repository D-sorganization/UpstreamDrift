"""Backward compatibility shim - module moved to gui_pkg.draggable_tabs."""

import sys as _sys

from .gui_pkg import draggable_tabs as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
