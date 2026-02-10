"""Backward compatibility shim - module moved to physics.ball_flight_physics."""

import sys as _sys

from .physics import ball_flight_physics as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
