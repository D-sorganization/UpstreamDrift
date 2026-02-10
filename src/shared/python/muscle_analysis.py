"""Backward compatibility shim - module moved to biomechanics.muscle_analysis."""

import sys as _sys

from .biomechanics import muscle_analysis as _real_module  # noqa: E402
from .biomechanics.muscle_analysis import (  # noqa: F401
    MuscleSynergyAnalyzer,
    SynergyResult,
    logger,
)

_sys.modules[__name__] = _real_module
