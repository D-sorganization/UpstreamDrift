"""Backward compatibility shim - module moved to physics.terrain_engine."""

import sys as _sys

from .physics import terrain_engine as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
