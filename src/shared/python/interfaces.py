"""Backward-compatible shim — canonical location: engine_core.interfaces."""

from src.shared.python.engine_core.interfaces import (  # noqa: F401
    PhysicsEngine,
    RecorderInterface,
)

__all__ = ["PhysicsEngine", "RecorderInterface"]
