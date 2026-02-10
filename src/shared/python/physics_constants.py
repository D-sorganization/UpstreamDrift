"""Backward compatibility shim - module moved to core.physics_constants."""

import sys as _sys

from .core import physics_constants as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
