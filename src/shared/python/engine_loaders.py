"""Backward-compatible shim — canonical location: engine_core.engine_loaders."""

from src.shared.python.engine_core.engine_loaders import *  # noqa: F401,F403
from src.shared.python.engine_core.engine_loaders import LOADER_MAP  # noqa: F401

__all__ = ["LOADER_MAP"]
