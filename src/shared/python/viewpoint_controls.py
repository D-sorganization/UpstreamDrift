"""Backward compatibility shim - module moved to gui_pkg.viewpoint_controls."""

import sys as _sys

from .gui_pkg import viewpoint_controls as _real_module  # noqa: E402
from .gui_pkg.viewpoint_controls import (  # noqa: F401
    DEFAULT_CAMERA_DISTANCE,
    DEFAULT_GOLFER_POSITION,
    DEFAULT_TARGET_DIRECTION,
    CameraPreset,
    CameraState,
    TrackingTarget,
    ViewpointController,
    ViewportLayout,
    compute_tracking_look_at,
    create_camera_from_preset,
    create_custom_camera,
    create_multiview_layout,
    create_standard_2x2_layout,
    create_transition_sequence,
    get_preset_camera_params,
    interpolate_camera_states,
    logger,
    spherical_to_cartesian,
)

_sys.modules[__name__] = _real_module
