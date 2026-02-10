"""Backward compatibility shim - module moved to core.numerical_constants."""

import sys as _sys

from .core import numerical_constants as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
