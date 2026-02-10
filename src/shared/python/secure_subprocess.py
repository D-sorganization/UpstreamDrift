"""Backward compatibility shim - module moved to security.secure_subprocess."""

import sys as _sys

from .security import secure_subprocess as _real_module  # noqa: E402
from .security.secure_subprocess import (  # noqa: F401
    ALLOWED_EXECUTABLES,
    ALLOWED_SCRIPT_DIRECTORIES,
    SecureSubprocessError,
    logger,
    secure_popen,
    secure_run,
    validate_executable,
    validate_script_path,
)

_sys.modules[__name__] = _real_module
