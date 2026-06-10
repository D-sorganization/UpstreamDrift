"""Backward-compatible shim — canonical location: engine_core.engine_loaders."""

from src.shared.python.engine_core.engine_loaders import (  # noqa: F401
    LOADER_MAP,
    load_matlab_2d_engine,
    load_matlab_3d_engine,
)

__all__ = ["LOADER_MAP", "load_matlab_2d_engine", "load_matlab_3d_engine"]
