"""Backward compatibility shim - module moved to physics.aerodynamics."""

import sys as _sys

from .physics import aerodynamics as _real_module  # noqa: E402
from .physics.aerodynamics import (  # noqa: F401
    AerodynamicsConfig,
    AerodynamicsEngine,
    DragModel,
    EnvironmentRandomizer,
    EnvironmentSnapshot,
    LiftModel,
    MagnusModel,
    RandomizationConfig,
    TurbulenceModel,
    WindConfig,
    WindGust,
    WindModel,
)

_sys.modules[__name__] = _real_module
