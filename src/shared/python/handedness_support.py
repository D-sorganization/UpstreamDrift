"""Backward compatibility shim - module moved to config.handedness_support."""

import sys as _sys

from .config import handedness_support as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
