"""Backwards-compatible shim for the relocated dataset extractor.

The implementation moved to
``src.shared.python.motion_matching.surrogate.perstep.extract_dataset`` per issue #4044.
"""

from __future__ import annotations

import warnings

from src.shared.python.motion_matching.surrogate.perstep.extract_dataset import *  # noqa: F401, F403

warnings.warn(
    "MachineLearning.extract_dynamics_dataset moved to "
    "src.shared.python.motion_matching.surrogate.perstep.extract_dataset (issue #4044). "  # noqa: E501
    "Update your imports; this shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
