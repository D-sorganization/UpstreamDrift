"""Backward compatibility shim - module moved to engine_core.checkpoint."""
import sys as _sys

from .engine_core import checkpoint as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
