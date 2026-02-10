"""Backward compatibility shim - module moved to biomechanics.swing_comparison."""

import sys as _sys

from .biomechanics import swing_comparison as _real_module  # noqa: E402
from .biomechanics.swing_comparison import (  # noqa: F401
    EPSILON,
    SIMILARITY_SCORE_CONSTANT,
    ComparisonMetric,
    DTWResult,
    SwingComparator,
)

_sys.modules[__name__] = _real_module
