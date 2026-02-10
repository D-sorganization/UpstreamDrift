"""Backward compatibility shim - module moved to gui_pkg.viewpoint_controls."""
import sys as _sys

from .gui_pkg import viewpoint_controls as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
