"""Backward-compatible shim — canonical location: engine_core.capabilities."""

from src.shared.python.engine_core.capabilities import (  # noqa: F401
    CapabilityLevel,
    EngineCapabilities,
)

__all__ = ["EngineCapabilities", "CapabilityLevel"]
