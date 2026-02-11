"""Engine loader functions — backward-compatible re-export.

The canonical location for engine loaders is now ``src.engines.loaders``.
This shim preserves the old import path so existing code (e.g.
``engine_manager.py``) continues to work without changes.

Migration:
    Old: from src.shared.python.engine_core.engine_loaders import LOADER_MAP
    New: from src.engines.loaders import LOADER_MAP
"""

from src.engines.loaders import (
    LOADER_MAP,
    load_drake_engine,
    load_mujoco_engine,
    load_myosim_engine,
    load_opensim_engine,
    load_pendulum_engine,
    load_pinocchio_engine,
    load_putting_green_engine,
)

__all__ = [
    "LOADER_MAP",
    "load_drake_engine",
    "load_mujoco_engine",
    "load_myosim_engine",
    "load_opensim_engine",
    "load_pendulum_engine",
    "load_pinocchio_engine",
    "load_putting_green_engine",
]
