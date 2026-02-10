"""Backward compatibility shim - module moved to biomechanics.swing_plane_visualization."""

import sys as _sys

from .biomechanics import swing_plane_visualization as _real_module  # noqa: E402
from .biomechanics.swing_plane_visualization import (  # noqa: F401
    DEFAULT_FSP_COLOR,
    DEFAULT_PLANE_COLOR,
    DEFAULT_PLANE_SIZE,
    DEFAULT_TRAJECTORY_COLOR,
    LOGGER,
    SwingPlaneScene,
    SwingPlaneVisualization,
    SwingPlaneVisualizer,
    TrajectoryVisualization,
    compute_trajectory_deviations,
    create_deviation_colormap,
    create_fsp_visualization,
    create_instantaneous_plane_visualization,
    generate_plane_vertices,
)

_sys.modules[__name__] = _real_module
