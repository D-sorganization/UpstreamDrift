"""Backward compatibility shim - module moved to gui_pkg.ellipsoid_visualization."""

import sys as _sys

from .gui_pkg import ellipsoid_visualization as _real_module  # noqa: E402
from .gui_pkg.ellipsoid_visualization import (  # noqa: F401
    FORCE_ELLIPSOID_COLOR,
    FORCE_ELLIPSOID_SINGULAR_SCALE,
    LOGGER,
    SINGULAR_ELLIPSOID_COLOR,
    SINGULAR_VALUE_TOLERANCE,
    VELOCITY_ELLIPSOID_COLOR,
    EllipsoidData,
    EllipsoidSequence,
    EllipsoidVisualizer,
    compute_force_ellipsoid,
    compute_velocity_ellipsoid,
    ellipsoid_to_json,
    export_ellipsoid_obj,
    export_ellipsoid_sequence_json,
    export_ellipsoid_stl,
    generate_ellipsoid_mesh,
)

_sys.modules[__name__] = _real_module
