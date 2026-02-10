"""Backward compatibility shim - module moved to core.type_utils."""
import sys as _sys

from .core import type_utils as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
