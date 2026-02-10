"""Backward compatibility shim - module moved to security.env_validator."""

import sys as _sys

from .security import env_validator as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
