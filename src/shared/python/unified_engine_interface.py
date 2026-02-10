"""Backward compatibility shim - module moved to engine_core.unified_engine_interface."""

import sys as _sys

from .engine_core import unified_engine_interface as _real_module  # noqa: E402
from .engine_core.unified_engine_interface import (  # noqa: F401
    UnifiedEngineInterface,
    create_unified_interface,
    logger,
    quick_setup,
)

_sys.modules[__name__] = _real_module
