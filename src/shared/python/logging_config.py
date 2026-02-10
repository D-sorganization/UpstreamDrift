"""Backward compatibility shim - module moved to logging_pkg.logging_config."""

import sys as _sys

from .logging_pkg import logging_config as _real_module  # noqa: E402
from .logging_pkg.logging_config import (  # noqa: F401
    DEFAULT_LOG_FORMAT,
    DETAILED_LOG_FORMAT,
    SIMPLE_LOG_FORMAT,
    LogLevel,
    add_file_handler,
    configure_gui_logging,
    configure_test_logging,
    get_logger,
    setup_logging,
)

_sys.modules[__name__] = _real_module
