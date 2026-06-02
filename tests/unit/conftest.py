"""Pytest configuration for unit tests.

Provides shared fixtures and setup for all unit tests.
"""

import sys
from unittest.mock import MagicMock

import pytest


def pytest_configure(config):
    """Configure pytest plugins and initialize fixtures early.

    This hook runs before test collection, allowing us to mock modules
    that test files will try to import. This prevents ModuleNotFoundError
    during collection phase.

    Using sys.modules directly here (not patch.dict) ensures the mock
    persists through collection. Individual test classes use @patch.dict
    to clean up per-test-scope.
    """
    # Mock Drake dependencies
    if "pydrake" not in sys.modules:
        sys.modules["pydrake"] = MagicMock()
    if "pydrake.all" not in sys.modules:
        sys.modules["pydrake.all"] = sys.modules["pydrake"]

    # Mock optimization dependencies
    if "casadi" not in sys.modules:
        sys.modules["casadi"] = MagicMock()
    if "pinocchio" not in sys.modules:
        sys.modules["pinocchio"] = MagicMock()
    if "pinocchio.casadi" not in sys.modules:
        sys.modules["pinocchio.casadi"] = MagicMock()


@pytest.fixture(autouse=True)
def _reset_mocks_between_tests():
    """Reset all mocked modules before each test to prevent cross-test pollution.

    Although class-level @patch.dict decorators handle scoping, this fixture
    ensures a clean slate for each test function that doesn't have explicit
    patch decorators.

    Crucially, it also clears the engine-availability memo cache. The
    ``engine_availability`` probe imports ``sys.modules["pinocchio"]`` (etc.)
    and caches the result; with a ``MagicMock`` installed during collection it
    would otherwise cache the engine as AVAILABLE and poison later tests that
    genuinely require the real bindings (see issue #7042). The probe now
    detects mocks and reports NOT_INSTALLED, but resetting the cache per test
    keeps that determination from leaking across the mock/unmock boundary.
    """
    # Reset the mocks to MagicMock() to clear any state
    for module_name in [
        "pydrake",
        "pydrake.all",
        "casadi",
        "pinocchio",
        "pinocchio.casadi",
    ]:
        if module_name in sys.modules:
            sys.modules[module_name] = MagicMock()

    # Drop any memoised availability verdict so each test re-probes against the
    # current sys.modules state rather than a stale cross-test result.
    try:
        from src.shared.python.engine_core.engine_availability import (
            reset_engine_status_cache,
        )

        reset_engine_status_cache()
    except ImportError:
        # engine_availability is an internal module; if it cannot be imported
        # the cache does not exist to poison, so there is nothing to reset.
        pass

    yield
