"""Backward compatibility shim - module moved to physics.energy_monitor."""

import sys as _sys

from .physics import energy_monitor as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
