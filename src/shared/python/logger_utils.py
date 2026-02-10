"""Backward compatibility shim - module moved to logging_pkg.logger_utils."""

import sys as _sys

from .logging_pkg import logger_utils as _real_module  # noqa: E402
from .logging_pkg.logger_utils import (  # noqa: F401
    DEFAULT_SEED,
    LOG_FORMAT,
    LOG_LEVEL,
    get_logger,
    log_execution_time,
    set_seeds,
    setup_logging,
)

_sys.modules[__name__] = _real_module
