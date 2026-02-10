"""Backward compatibility shim - module moved to validation_pkg.comparative_plotting."""
import sys as _sys

from .validation_pkg import comparative_plotting as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
