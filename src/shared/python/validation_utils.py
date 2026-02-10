"""Backward compatibility shim - module moved to validation_pkg.validation_utils."""

import sys as _sys

from .validation_pkg import validation_utils as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
