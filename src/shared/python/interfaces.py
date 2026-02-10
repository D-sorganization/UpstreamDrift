"""Backward compatibility shim - module moved to engine_core.interfaces."""

import sys as _sys

from .engine_core import interfaces as _real_module  # noqa: E402
from .engine_core.interfaces import (  # noqa: F401
    PhysicsEngine,
    RecorderInterface,
)

_sys.modules[__name__] = _real_module
