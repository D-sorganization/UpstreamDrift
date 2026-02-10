"""Backward compatibility shim - module moved to physics.flight_model_options."""

import sys as _sys

from .physics import flight_model_options as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
