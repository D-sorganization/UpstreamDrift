"""Backward compatibility shim - module moved to physics.topography."""
import sys as _sys

from .physics import topography as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
