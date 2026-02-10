"""Backward compatibility shim - module moved to core.exceptions."""

import sys as _sys

from .core import exceptions as _real_module  # noqa: E402
from .core.exceptions import (  # noqa: F401
    ArrayDimensionError,
    DataFormatError,
    EngineNotAvailableError,
    EngineNotFoundError,
    GolfModelingError,
    GolfSuiteError,
    ValidationConstraintError,
    ValidationError,
)

_sys.modules[__name__] = _real_module
