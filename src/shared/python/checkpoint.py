"""Backward compatibility shim - module moved to engine_core.checkpoint."""

import sys as _sys

from .engine_core import checkpoint as _real_module  # noqa: E402
from .engine_core.checkpoint import (  # noqa: F401
    Checkpointable,
    CheckpointManager,
    StateCheckpoint,
    logger,
)

_sys.modules[__name__] = _real_module
