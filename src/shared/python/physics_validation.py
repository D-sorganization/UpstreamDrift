"""Backward compatibility shim - module moved to physics.physics_validation."""

import sys as _sys

from .physics import physics_validation as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
