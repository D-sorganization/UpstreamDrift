"""Pytest configuration for unit tests.

Provides shared fixtures and setup for all unit tests.
"""

import sys
from unittest.mock import MagicMock

import pytest

_MOCKED_MODULES: set[str] = set()


def pytest_configure(config):
    """Configure pytest plugins and initialize fixtures early.

    This hook runs before test collection, allowing us to mock modules
    that test files will try to import. This prevents ModuleNotFoundError
    during collection phase.

    Using sys.modules directly here (not patch.dict) ensures the mock
    persists through collection. Individual test classes use @patch.dict
    to clean up per-test-scope.
    """
    # Mock Drake dependencies if not installed
    if "pydrake" not in sys.modules:
        try:
            import pydrake.all  # noqa: F401

            if type(sys.modules.get("pydrake.all")).__module__ == "unittest.mock":
                raise ImportError("Mocked")
        except (ImportError, RuntimeError, AttributeError):
            sys.modules["pydrake"] = MagicMock()
            sys.modules["pydrake.all"] = sys.modules["pydrake"]
            _MOCKED_MODULES.update({"pydrake", "pydrake.all"})

    # Mock optimization dependencies if not installed
    if "casadi" not in sys.modules:
        try:
            import casadi  # noqa: F401

            if type(sys.modules.get("casadi")).__module__ == "unittest.mock":
                raise ImportError("Mocked")
        except (ImportError, RuntimeError, AttributeError):
            sys.modules["casadi"] = MagicMock()
            _MOCKED_MODULES.add("casadi")
    if "pinocchio" not in sys.modules:
        try:
            import pinocchio  # noqa: F401

            if type(sys.modules.get("pinocchio")).__module__ == "unittest.mock":
                raise ImportError("Mocked")
        except (ImportError, RuntimeError, AttributeError):
            sys.modules["pinocchio"] = MagicMock()
            _MOCKED_MODULES.add("pinocchio")
    if "pinocchio.casadi" not in sys.modules:
        try:
            import pinocchio.casadi  # noqa: F401

            if type(sys.modules.get("pinocchio.casadi")).__module__ == "unittest.mock":
                raise ImportError("Mocked")
        except (ImportError, RuntimeError, AttributeError):
            sys.modules["pinocchio.casadi"] = MagicMock()
            _MOCKED_MODULES.add("pinocchio.casadi")


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
    # Reset only the modules that were actually mocked
    for module_name in _MOCKED_MODULES:
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
