"""Backward compatibility shim - module moved to gui_pkg.image_utils."""
import sys as _sys

from .gui_pkg import image_utils as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
