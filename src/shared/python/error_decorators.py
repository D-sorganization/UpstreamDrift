"""Backward compatibility shim - module moved to core.error_decorators."""

import sys as _sys

from .core import error_decorators as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
