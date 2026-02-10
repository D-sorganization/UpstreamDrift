"""Backward compatibility shim - module moved to validation_pkg.statistical_analysis."""
import sys as _sys

from .validation_pkg import statistical_analysis as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
