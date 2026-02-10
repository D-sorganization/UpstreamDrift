"""Backward compatibility shim - module moved to gui_pkg.launcher_utils."""

import sys as _sys

from .gui_pkg import launcher_utils as _real_module  # noqa: E402
from .gui_pkg.launcher_utils import (  # noqa: F401
    check_python_dependencies,
    ensure_environment_var,
    get_repo_root,
    git_sync_repository,
    invoke_main,
    logger,
)

_sys.modules[__name__] = _real_module
