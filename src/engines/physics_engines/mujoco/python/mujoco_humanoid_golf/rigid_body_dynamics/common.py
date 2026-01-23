"""Common constants and utilities for rigid body dynamics algorithms.

This module provides shared constants and helper functions used by RNEA, ABA,
and CRBA algorithms to prevent code duplication and ensure consistency.
"""

from __future__ import annotations

import numpy as np

from src.shared.python import constants

# Default gravity vector (spatial acceleration)
# -9.81 m/s^2 in z-direction (standard earth gravity)
# Format: [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]
DEFAULT_GRAVITY = np.array([0, 0, 0, 0, 0, -constants.GRAVITY_M_S2])
DEFAULT_GRAVITY.flags.writeable = False

# Optimized negative default gravity for RNEA/ABA to avoid allocation
NEG_DEFAULT_GRAVITY = -DEFAULT_GRAVITY
NEG_DEFAULT_GRAVITY.flags.writeable = False
