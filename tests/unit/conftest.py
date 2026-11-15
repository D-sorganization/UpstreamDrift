"""Pytest configuration for unit tests.

Provides shared fixtures and setup for all unit tests.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

_MOCKED_MODULES: set[str] = set()
_UNIT_ROOT = Path(__file__).resolve().parent


def _is_mock_module(module: object | None) -> bool:
    """Return True if *module* is a unittest.mock stand-in rather than real."""
    return module is not None and type(module).__module__ == "unittest.mock"


def _reset_engine_availability_cache() -> None:
    """Clear memoized engine availability so mock/real transitions re-probe."""
    try:
        from src.shared.python.engine_core.engine_availability import (
            reset_engine_status_cache,
        )

        reset_engine_status_cache()
    except ImportError:
        # engine_availability is an internal module; if it cannot be imported
        # the cache does not exist to poison, so there is nothing to reset.
        pass


def _is_unit_item(item: pytest.Item) -> bool:
    """True when *item* belongs under the tests/unit/ directory tree."""
    try:
        return Path(str(item.path)).resolve().is_relative_to(_UNIT_ROOT)
    except (AttributeError, OSError, ValueError):
        return False


def _install_unit_mocks() -> None:
    """Ensure every module stubbed at collection time has a fresh MagicMock."""
    for module_name in _MOCKED_MODULES:
        sys.modules[module_name] = MagicMock()
    _reset_engine_availability_cache()


def _remove_unit_mocks() -> None:
    """Remove unit-only MagicMock stubs from sys.modules when outside tests/unit."""
    removed = False
    for module_name in _MOCKED_MODULES:
        if _is_mock_module(sys.modules.get(module_name)):
            sys.modules.pop(module_name, None)
            removed = True
    if removed:
        _reset_engine_availability_cache()


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


def pytest_collection_finish(session: pytest.Session) -> None:
    """Drop collection-time stubs before non-unit tests begin."""
    del session
    _remove_unit_mocks()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Ensure unit stubs are active only while executing tests/unit/ items."""
    if _is_unit_item(item):
        _install_unit_mocks()
    else:
        _remove_unit_mocks()


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    """Strip unit-only MagicMocks as execution leaves tests/unit/."""
    if _is_unit_item(item) and (nextitem is None or not _is_unit_item(nextitem)):
        _remove_unit_mocks()


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
    _install_unit_mocks()

    yield
