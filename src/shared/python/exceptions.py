"""Backward-compatible shim — canonical location: core.exceptions."""

from src.shared.python.core.exceptions import (
    EngineNotFoundError,
    GolfModelingError,
)

__all__ = ["GolfModelingError", "EngineNotFoundError"]
