"""Backward compatibility shim - module moved to validation_pkg.validation_helpers."""

import sys as _sys

from .validation_pkg import validation_helpers as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
