"""Backward compatibility shim - module moved to biomechanics.kinematic_sequence."""

import sys as _sys

from .biomechanics import kinematic_sequence as _real_module  # noqa: E402
from .biomechanics.kinematic_sequence import (  # noqa: F401
    KinematicSequenceAnalyzer,
    KinematicSequenceResult,
    SegmentPeak,
    SegmentTimingAnalyzer,
    SegmentTimingResult,
)

_sys.modules[__name__] = _real_module
