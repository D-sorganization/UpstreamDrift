"""Backward compatibility shim - module moved to engine_core.mock_engine."""

import sys as _sys

from .engine_core import mock_engine as _real_module  # noqa: E402
from .engine_core.mock_engine import (  # noqa: F401
    MockPhysicsEngine,
    get_mock_engine,
    logger,
)

_sys.modules[__name__] = _real_module
