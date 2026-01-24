"""Unified Exception Hierarchy for Golf Modeling Suite.

DEPRECATED: This module is maintained for backwards compatibility.
New code should import exceptions directly from error_utils.py.

This module redirects to src.shared.python.error_utils which contains
the consolidated exception hierarchy.

Migration:
    # Old import (deprecated):
    from src.shared.python.exceptions import GolfModelingError, EngineNotFoundError

    # New import (preferred):
    from src.shared.python.error_utils import GolfSuiteError, EngineNotAvailableError
"""

import warnings

from .error_utils import (
    DataFormatError,
    EngineNotAvailableError,
    GolfSuiteError,
    ValidationError,
)

# Re-export with old names for backwards compatibility
GolfModelingError = GolfSuiteError
EngineNotFoundError = EngineNotAvailableError
ValidationConstraintError = ValidationError


class ArrayDimensionError(GolfSuiteError):
    """Raised when an array has incorrect dimensions.

    This exception is specific to array dimension validation.
    """

    def __init__(
        self,
        array_name: str = "array",
        expected_shape: tuple[int, ...] | None = None,
        actual_shape: tuple[int, ...] | None = None,
        message: str | None = None,
    ):
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape

        if message:
            full_message = message
        else:
            full_message = f"Array '{array_name}' has incorrect dimensions"
            if expected_shape and actual_shape:
                full_message += (
                    f". Expected shape: {expected_shape}, got: {actual_shape}"
                )
            elif expected_shape:
                full_message += f". Expected shape: {expected_shape}"
            elif actual_shape:
                full_message += f". Got shape: {actual_shape}"

        super().__init__(full_message)


# Export all for backwards compatibility
__all__ = [
    # New names (preferred)
    "GolfSuiteError",
    "EngineNotAvailableError",
    "DataFormatError",
    "ValidationError",
    "ArrayDimensionError",
    # Old names (deprecated, for backwards compatibility)
    "GolfModelingError",
    "EngineNotFoundError",
    "ValidationConstraintError",
]


def __getattr__(name: str) -> type[Exception]:
    """Emit deprecation warnings for old exception names."""
    deprecated_names = {
        "GolfModelingError": "GolfSuiteError",
        "EngineNotFoundError": "EngineNotAvailableError",
        "ValidationConstraintError": "ValidationError",
    }
    if name in deprecated_names:
        warnings.warn(
            f"{name} is deprecated, use {deprecated_names[name]} from error_utils.py instead",
            DeprecationWarning,
            stacklevel=2,
        )
        result: type[Exception] = globals()[name]
        return result
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
