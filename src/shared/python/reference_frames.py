"""Backward compatibility shim - module moved to spatial_algebra.reference_frames."""

import sys as _sys

from .spatial_algebra import reference_frames as _real_module  # noqa: E402
from .spatial_algebra.reference_frames import (  # noqa: F401
    GEOMETRIC_TOLERANCE,
    ReferenceFrame,
    ReferenceFrameTransformer,
    SwingPlaneFrame,
    WrenchInFrame,
    compute_rotation_matrix_from_axes,
    decompose_wrench_in_swing_plane,
    fit_functional_swing_plane,
    fit_instantaneous_swing_plane,
    logger,
    transform_wrench_to_frame,
)

_sys.modules[__name__] = _real_module
