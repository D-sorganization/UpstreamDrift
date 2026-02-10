"""Backward compatibility shim - module moved to gui_pkg.help_system."""

import sys as _sys

from .gui_pkg import help_system as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
