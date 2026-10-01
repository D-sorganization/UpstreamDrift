"""Fixtures for unit tests in tests/unit/api."""

from __future__ import annotations

from typing import Iterator
import pytest


@pytest.fixture(autouse=True)
def _reset_api_rate_limiter() -> Iterator[None]:
    """Reset the SlowAPI rate limiter between tests to prevent spurious 429s in unit tests."""
    try:
        from src.api.rate_limit import limiter

        limiter.reset()
    except (ImportError, AttributeError, RuntimeError):
        pass
    yield
    try:
        from src.api.rate_limit import limiter

        limiter.reset()
    except (ImportError, AttributeError, RuntimeError):
        pass
