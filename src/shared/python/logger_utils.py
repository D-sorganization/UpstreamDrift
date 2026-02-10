"""Backward compatibility shim - module moved to logging_pkg.logger_utils."""
import sys as _sys

from .logging_pkg import logger_utils as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
