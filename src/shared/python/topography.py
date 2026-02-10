"""Backward compatibility shim - module moved to physics.topography."""

import sys as _sys

from .physics import topography as _real_module  # noqa: E402
from .physics.topography import (  # noqa: F401
    ElevationPoint,
    TopographyBounds,
    TopographyData,
    TopographyProvider,
    create_flat_terrain,
    create_sloped_terrain,
    create_undulating_terrain,
)

_sys.modules[__name__] = _real_module
