"""Backward compatibility shim - module moved to physics.terrain_mixin."""

import sys as _sys

from .physics import terrain_mixin as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
