"""Backward compatibility shim - module moved to physics.terrain_mixin."""

import sys as _sys

from .physics import terrain_mixin as _real_module  # noqa: E402
from .physics.terrain_mixin import (  # noqa: F401
    HasPosition,
    TerrainAwareSimulation,
    TerrainMixin,
    logger,
)

_sys.modules[__name__] = _real_module
