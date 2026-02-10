"""Backward compatibility shim - module moved to engine_core.engine_registry."""

import sys as _sys

from .engine_core import engine_registry as _real_module  # noqa: E402
from .engine_core.engine_registry import (  # noqa: F401
    EngineFactory,
    EngineRegistration,
    EngineRegistry,
    EngineStatus,
    EngineType,
    get_registry,
)

_sys.modules[__name__] = _real_module
