"""Backward compatibility shim - module moved to security.secure_subprocess."""
import sys as _sys

from .security import secure_subprocess as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
