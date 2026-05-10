"""Scientific computing package for trajectory analysis and simulation.

This package provides utilities for scientific computing with a focus on
reproducibility, proper documentation, and adherence to quality standards.
"""

import contextlib

__version__ = "1.0.0"
__author__ = "Scientific Computing Team"
__email__ = "team@example.com"


# Export commonly used functions and constants
from src.shared.python.logging_pkg.logger_utils import (
    get_logger,
    set_seeds,
    setup_logging,
)

# Register the Drake dashboard embed adapter with the launcher's
# embeddable-tool registry on import. Guarded by
# ``contextlib.suppress(ImportError)`` so importing this package keeps
# working in headless contexts where PyQt6 or the ``pydrake`` wheel is
# unavailable. Subtask 5 / #4998 of EPIC #4993.
with contextlib.suppress(ImportError):
    from . import _embed_adapter  # noqa: F401

__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "get_logger",
    "set_seeds",
    "setup_logging",
]
