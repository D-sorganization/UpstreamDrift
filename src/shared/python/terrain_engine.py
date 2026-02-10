"""Backward compatibility shim - module moved to physics.terrain_engine."""

import sys as _sys

from .physics import terrain_engine as _real_module  # noqa: E402
from .physics.terrain_engine import (  # noqa: F401
    CompressibleTurfModel,
    PhysicsEngineProtocol,
    TerrainAwareEngine,
    TerrainContactModel,
    TerrainGeometryGenerator,
    apply_terrain_to_engine,
    logger,
    register_terrain_parameters,
    validate_terrain,
)

_sys.modules[__name__] = _real_module
