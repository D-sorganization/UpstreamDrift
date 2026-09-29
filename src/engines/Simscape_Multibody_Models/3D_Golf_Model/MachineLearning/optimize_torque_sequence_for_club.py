"""Backwards-compatible shim for the relocated per-step torque optimizer.

The implementation moved to
``src.shared.python.motion_matching.surrogate.perstep.optimize`` per issue #4044.
"""

from __future__ import annotations

import warnings

from src.shared.python.motion_matching.surrogate.perstep.optimize import *  # noqa: F401, F403

warnings.warn(
    "MachineLearning.optimize_torque_sequence_for_club moved to "
    "src.shared.python.motion_matching.surrogate.perstep.optimize (issue #4044). "  # noqa: E501
    "Update your imports; this shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
