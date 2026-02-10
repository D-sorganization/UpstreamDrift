"""Backward compatibility shim - module moved to validation_pkg.comparative_analysis."""

import sys as _sys

from .validation_pkg import comparative_analysis as _real_module  # noqa: E402
from .validation_pkg.comparative_analysis import (  # noqa: F401
    AlignedSignals,
    ComparativeSwingAnalyzer,
    ComparisonMetric,
    RecorderInterface,
)

_sys.modules[__name__] = _real_module
