"""Backward compatibility shim - module moved to engine_core.engine_loaders."""

import sys as _sys

from .engine_core import engine_loaders as _real_module  # noqa: E402
from .engine_core.engine_loaders import (  # noqa: F401
    LOADER_MAP,
    load_drake_engine,
    load_mujoco_engine,
    load_myosim_engine,
    load_opensim_engine,
    load_pendulum_engine,
    load_pinocchio_engine,
    load_putting_green_engine,
)

_sys.modules[__name__] = _real_module
