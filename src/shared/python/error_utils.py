"""Backward compatibility shim - module moved to core.error_utils."""

import sys as _sys

from .core import error_utils as _real_module  # noqa: E402
from .core.error_utils import (  # noqa: F401
    ConfigurationError,
    ContractViolationError,
    DataFormatError,
    EngineNotAvailableError,
    EnvironmentError,
    EnvironmentValidationError,
    FileNotFoundError_,
    FileNotFoundIOError,
    FileOperationError,
    FileParseError,
    GolfSuiteError,
    InvariantError,
    IOError,
    IOUtilsError,
    ModelError,
    PhysicalValidationError,
    PostconditionError,
    PreconditionError,
    ResourceError,
    SimulationError,
    StateError,
    TimeoutError,
    ValidationError,
    format_file_error,
    format_import_error,
    format_range_error,
    format_type_error,
    format_validation_error,
    handle_import_error,
)

_sys.modules[__name__] = _real_module
