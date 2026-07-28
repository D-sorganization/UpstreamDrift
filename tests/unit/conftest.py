"""Pytest configuration for unit tests.

Provides shared fixtures and setup for all unit tests.
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_engine_availability_cache_between_tests():
    """Reset optional-engine availability memoization around each unit test."""
    _reset_engine_status_cache_if_available()
    yield
    _reset_engine_status_cache_if_available()


def _reset_engine_status_cache_if_available() -> None:
    """Drop memoized engine availability verdicts when the helper is importable."""
    try:
        from src.shared.python.engine_core.engine_availability import (
            reset_engine_status_cache,
        )

        reset_engine_status_cache()
    except ImportError:
        # engine_availability is an internal module; if it cannot be imported
        # the cache does not exist to poison, so there is nothing to reset.
        pass
