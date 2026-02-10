"""Backward compatibility shim - module moved to gui_pkg.launcher_utils."""
import sys as _sys

from .gui_pkg import launcher_utils as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
