"""Backward compatibility shim - module moved to engine_core.capabilities."""

import sys as _sys

from .engine_core import capabilities as _real_module  # noqa: E402
from .engine_core.capabilities import (  # noqa: F401
    CapabilityLevel,
    EngineCapabilities,
    logger,
)

_sys.modules[__name__] = _real_module
