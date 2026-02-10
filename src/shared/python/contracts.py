"""Backward compatibility shim - module moved to core.contracts."""
import sys as _sys

from .core import contracts as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
