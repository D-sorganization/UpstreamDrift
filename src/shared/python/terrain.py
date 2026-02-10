"""Backward compatibility shim - module moved to physics.terrain."""

import sys as _sys

from .physics import terrain as _real_module  # noqa: E402
from .physics.terrain import (  # noqa: F401
    MATERIALS,
    TERRAIN_MATERIAL_MAP,
    ElevationMap,
    SurfaceMaterial,
    Terrain,
    TerrainConfig,
    TerrainPatch,
    TerrainRegion,
    TerrainType,
    compute_gravity_on_slope,
    compute_roll_direction,
    create_flat_terrain,
    create_sloped_terrain,
    create_terrain_from_config,
    get_contact_normal,
    logger,
)

_sys.modules[__name__] = _real_module
