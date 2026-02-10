"""Backward compatibility shim - module moved to engine_core.unified_engine_interface."""
import sys as _sys

from .engine_core import unified_engine_interface as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
