"""Backward compatibility shim - module moved to security.security_utils."""
import sys as _sys

from .security import security_utils as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
