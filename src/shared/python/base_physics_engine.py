"""Backward compatibility shim - module moved to engine_core.base_physics_engine."""

import sys as _sys

from .engine_core import base_physics_engine as _real_module  # noqa: E402
from .engine_core.base_physics_engine import (  # noqa: F401
    BasePhysicsEngine,
    EngineState,
    ModelLoadingMixin,
    SimulationMixin,
    logger,
)

_sys.modules[__name__] = _real_module
