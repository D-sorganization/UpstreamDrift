"""Backward compatibility shim - module moved to physics.flight_model_options."""

import sys as _sys

from .physics import flight_model_options as _real_module  # noqa: E402
from .physics.flight_model_options import (  # noqa: F401
    DEFAULT_OPTIONS,
    FlightModelOptions,
    compute_air_density_at_altitude,
    compute_spin_decay,
)

_sys.modules[__name__] = _real_module
