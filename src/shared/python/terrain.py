"""Backward compatibility shim - module moved to physics.terrain."""
import sys as _sys

from .physics import terrain as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
