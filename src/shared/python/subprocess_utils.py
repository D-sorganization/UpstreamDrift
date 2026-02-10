"""Backward compatibility shim - module moved to security.subprocess_utils."""

import sys as _sys

from .security import subprocess_utils as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
