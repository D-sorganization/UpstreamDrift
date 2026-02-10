"""Backward compatibility shim - module moved to gui_pkg.help_content."""
import sys as _sys

from .gui_pkg import help_content as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
