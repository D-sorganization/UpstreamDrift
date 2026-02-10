"""Backward compatibility shim - module moved to core.exceptions."""
import sys as _sys

from .core import exceptions as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
