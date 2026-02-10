"""Backward compatibility shim - module moved to engine_core.base_physics_engine."""

import sys as _sys

from .engine_core import base_physics_engine as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
