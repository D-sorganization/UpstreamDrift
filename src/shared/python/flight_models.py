"""Backward compatibility shim - module moved to physics.flight_models."""
import sys as _sys

from .physics import flight_models as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
