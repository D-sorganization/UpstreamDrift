"""Backward compatibility shim - module moved to physics.energy_monitor."""

import sys as _sys

from .physics import energy_monitor as _real_module  # noqa: E402
from .physics.energy_monitor import (  # noqa: F401
    ENERGY_DRIFT_CRITICAL_PCT,
    ENERGY_DRIFT_TOLERANCE_PCT,
    ConservationMonitor,
    EnergySnapshot,
    IntegrationFailureError,
    logger,
)

_sys.modules[__name__] = _real_module
