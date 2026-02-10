"""Backward compatibility shim - module moved to biomechanics.multi_muscle."""

import sys as _sys

from .biomechanics import multi_muscle as _real_module  # noqa: E402
from .biomechanics.multi_muscle import (  # noqa: F401
    AntagonistPair,
    MuscleAttachment,
    MuscleGroup,
    create_elbow_muscle_system,
    logger,
)

_sys.modules[__name__] = _real_module
