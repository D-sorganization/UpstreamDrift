"""Backward compatibility shim - module moved to biomechanics.myosuite_adapter."""

import sys as _sys

from .biomechanics import myosuite_adapter as _real_module  # noqa: E402
from .biomechanics.myosuite_adapter import (  # noqa: F401
    MuscleDrivenEnv,
    logger,
    train_muscle_policy,
)

_sys.modules[__name__] = _real_module
