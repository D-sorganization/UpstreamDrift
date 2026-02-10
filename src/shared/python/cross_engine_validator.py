"""Backward compatibility shim - module moved to engine_core.cross_engine_validator."""

import sys as _sys

from .engine_core import cross_engine_validator as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
