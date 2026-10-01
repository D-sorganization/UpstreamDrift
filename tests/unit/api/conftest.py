"""Fixtures for unit tests in tests/unit/api."""

from __future__ import annotations

from typing import Iterator
import pytest


@pytest.fixture(autouse=True)
def _reset_api_rate_limiter() -> Iterator[None]:
    """Disable SlowAPI rate limiter and clear storage between unit tests to prevent spurious 429s."""
    prev_enabled = True
    try:
        from src.api.rate_limit import limiter

        prev_enabled = getattr(limiter, "enabled", True)
        limiter.enabled = False
        storage = getattr(limiter, "_storage", None)
        if storage is not None:
            if hasattr(storage, "storage") and hasattr(storage.storage, "clear"):
                storage.storage.clear()
            if hasattr(storage, "events") and hasattr(storage.events, "clear"):
                storage.events.clear()
    except (ImportError, AttributeError, RuntimeError):
        pass
    yield
    try:
        from src.api.rate_limit import limiter

        limiter.enabled = prev_enabled
    except (ImportError, AttributeError, RuntimeError):
        pass
