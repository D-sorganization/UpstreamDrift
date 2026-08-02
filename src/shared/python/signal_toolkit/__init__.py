# mypy: ignore-errors
"""Signal Processing Toolkit - Comprehensive Signal Analysis Library.

A production-ready signal processing library for generating, fitting,
filtering, and analyzing signals.

Features:
    - Signal Generation: 13 signal types (sine, cosine, chirp, etc.)
    - Function Fitting: Sinusoid, exponential, linear, polynomial, custom
    - Digital Filters: Butterworth, Chebyshev, Bessel, adaptive (LMS/RLS)
    - Calculus: Differentiation, integration, tangent lines, curvature
    - Series: Taylor/Maclaurin series expansions with convergence analysis
    - Noise Generation: White, pink, brown, blue, violet, impulse
    - Limits: Saturation, rate limiting, deadband, hysteresis, backlash
    - I/O: CSV, JSON, NPZ, MAT, numpy array support
    - Visualization: PyQt6 widget for interactive signal analysis

Version: 2.1.0

Lazy-loading strategy (issue #1696 - god module refactor):
    All submodule imports are deferred to first attribute access via
    ``__getattr__``.  The dispatch table has been moved to ``_lazy_map.py``
    to keep this file below 120 lines.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from src.shared.python.signal_toolkit._lazy_map import LAZY

logger = logging.getLogger(__name__)

__version__ = "2.1.0"

# ---------------------------------------------------------------------------
# Optional imports (PyQt6 / matplotlib)
# ---------------------------------------------------------------------------

try:
    from .polynomial_generator import PolynomialGeneratorWidget

    HAS_POLYNOMIAL_GENERATOR = True
except ImportError:
    PolynomialGeneratorWidget = None
    HAS_POLYNOMIAL_GENERATOR = False
    logger.debug("PolynomialGeneratorWidget not available (requires PyQt6)")

try:
    from .widget import SignalToolkitWidget

    HAS_WIDGET = True
except ImportError:
    SignalToolkitWidget = None
    HAS_WIDGET = False
    logger.debug("SignalToolkitWidget not available (requires PyQt6 + matplotlib)")


def __getattr__(name: str) -> Any:
    """Lazy attribute loader - defers submodule imports to first access."""
    if name in LAZY:
        module_path, attr = LAZY[name]
        mod = importlib.import_module(module_path, package=__name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *LAZY.keys(),
    "PolynomialGeneratorWidget",
    "SignalToolkitWidget",
    "HAS_POLYNOMIAL_GENERATOR",
    "HAS_WIDGET",
]
