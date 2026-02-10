"""Backward compatibility shim - module moved to security.subprocess_utils."""

import sys as _sys

from .security import subprocess_utils as _real_module  # noqa: E402
from .security.subprocess_utils import (  # noqa: F401
    DEFAULT_SUBPROCESS_TIMEOUT,
    CommandRunner,
    ProcessManager,
    kill_process_tree,
    logger,
    run_command,
)

_sys.modules[__name__] = _real_module
