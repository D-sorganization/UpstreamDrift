"""Backward compatibility shim - module moved to core.constants."""
import sys as _sys

from .core import constants as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
