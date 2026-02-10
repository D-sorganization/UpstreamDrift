"""Backward compatibility shim - module moved to engine_core.engine_manager."""
import sys as _sys

from .engine_core import engine_manager as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
