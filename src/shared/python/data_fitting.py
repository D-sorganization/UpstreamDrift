"""Backward compatibility shim - module moved to validation_pkg.data_fitting."""
import sys as _sys

from .validation_pkg import data_fitting as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
