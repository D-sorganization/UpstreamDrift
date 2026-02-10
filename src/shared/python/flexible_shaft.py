"""Backward compatibility shim - module moved to physics.flexible_shaft."""

import sys as _sys

from .physics import flexible_shaft as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
