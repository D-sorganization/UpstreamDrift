"""Lazy and fallback service helpers for the local API server."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class LazyServiceProxy:
    """Lazily instantiate heavy API services on first attribute access."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        """init."""
        self._factory = factory
        self._service: Any | None = None

    def _resolve(self) -> Any:
        """resolve."""
        if self._service is None:
            self._service = self._factory()
        return self._service

    def __getattr__(self, name: str) -> Any:
        """getattr."""
        return getattr(self._resolve(), name)


class UnavailableEngineManager:
    """Engine manager fallback used when optional engine imports are unavailable."""

    def __init__(self, reason: str) -> None:
        """init."""
        self.reason = reason

    def get_available_engines(self) -> list[Any]:
        """get available engines."""
        return []

    def get_current_engine(self) -> None:
        """get current engine."""
        return

    def get_engine_status(self, _engine_type: Any) -> Any:
        """get engine status."""
        from src.shared.python.engine_core.engine_registry import EngineStatus

        return EngineStatus.UNAVAILABLE

    def get_active_physics_engine(self) -> None:
        """get active physics engine."""
        return
