"""Backwards-compatible shim for the relocated per-step trainer.

The implementation moved to
``src.shared.python.motion_matching.surrogate.perstep.train`` per issue #4044.
This shim re-exports the public symbols and emits a ``DeprecationWarning`` on
import so existing scripts that ``from train_dynamics_surrogate import ...``
keep working for one release while callers migrate.
"""

from __future__ import annotations

import warnings

from src.shared.python.motion_matching.surrogate.perstep.train import *  # noqa: F401, F403
from src.shared.python.motion_matching.surrogate.perstep.train import (  # noqa: F401
    DynamicsMLP,
    TrainConfig,
    main,
)

warnings.warn(
    "MachineLearning.train_dynamics_surrogate moved to "
    "src.shared.python.motion_matching.surrogate.perstep.train (issue #4044). "  # noqa: E501
    "Update your imports; this shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
