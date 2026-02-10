"""Backward compatibility shim - module moved to config.environment."""
import sys as _sys

from .config import environment as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
