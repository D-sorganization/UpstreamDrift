"""Backward compatibility shim - module moved to config.handedness_support."""

import sys as _sys

from .config import handedness_support as _real_module  # noqa: E402
from .config.handedness_support import (  # noqa: F401
    SAGITTAL_MIRROR,
    Y_AXIS_ALIGNMENT_THRESHOLD,
    Y_AXIS_SIGNIFICANCE_THRESHOLD,
    Handedness,
    HandednessConverter,
    MirrorTransform,
    create_mirror_transform,
    detect_handedness_from_metadata,
    logger,
    mirror_angular_velocity,
    mirror_joint_configuration,
    mirror_position,
    mirror_rotation_matrix,
    mirror_trajectory,
    mirror_velocity,
    validate_energy_conservation,
    validate_mirror_trajectory,
)

_sys.modules[__name__] = _real_module
