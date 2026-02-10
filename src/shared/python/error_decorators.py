"""Backward compatibility shim - module moved to core.error_decorators."""

import sys as _sys

from .core import error_decorators as _real_module  # noqa: E402
from .core.error_decorators import (  # noqa: F401
    ErrorContext,
    F,
    check_module_available,
    handle_import_error,
    log_errors,
    logger,
    retry_on_error,
    safe_import,
    validate_args,
)

_sys.modules[__name__] = _real_module
