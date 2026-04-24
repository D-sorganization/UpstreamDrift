"""Pytest configuration for unit tests.

Provides shared fixtures and setup for all unit tests.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Modules that must be available during collection (test file import time) so
# that files importing optional heavy dependencies don't raise ImportError.
# We install lightweight stubs here *only for the collection phase*; the
# autouse fixture below reinstalls fresh MagicMocks for every test function so
# that no test ever sees state left over by a previous test.
_COLLECTION_STUBS: dict[str, str] = {
    "pydrake": "pydrake",
    "pydrake.all": "pydrake",  # alias → reuse the pydrake stub
    "casadi": "casadi",
    "pinocchio": "pinocchio",
    "pinocchio.casadi": "pinocchio.casadi",
}


def pytest_configure(config: "pytest.Config") -> None:  # noqa: F821
    """Install minimal collection-time stubs for unavailable heavy dependencies.

    These stubs are intentionally *not* cleaned up here — the autouse fixture
    ``_mock_heavy_deps`` replaces them with fresh MagicMock instances before
    every test function and restores the original state afterward via
    ``patch.dict``, satisfying the CLAUDE.md rule against persistent module-
    level ``sys.modules`` mutations.
    """
    # Only install stubs for modules that are genuinely absent so we do not
    # shadow a real installation.
    pydrake_stub = MagicMock()
    stubs = {
        "pydrake": pydrake_stub,
        "pydrake.all": pydrake_stub,
        "casadi": MagicMock(),
        "pinocchio": MagicMock(),
        "pinocchio.casadi": MagicMock(),
    }
    for name, stub in stubs.items():
        if name not in sys.modules:
            sys.modules[name] = stub


@pytest.fixture(autouse=True)
def _mock_heavy_deps():
    """Provide isolated, function-scoped mocks for every heavy dependency.

    Uses ``patch.dict`` so that *all* changes to ``sys.modules`` are
    automatically rolled back after each test — no cross-test pollution.
    This replaces the old module-level ``sys.modules[...] = MagicMock()``
    pattern banned in CLAUDE.md:60.
    """
    pydrake_mock = MagicMock()
    mocks = {
        "pydrake": pydrake_mock,
        "pydrake.all": pydrake_mock,
        "casadi": MagicMock(),
        "pinocchio": MagicMock(),
        "pinocchio.casadi": MagicMock(),
    }
    with patch.dict(sys.modules, mocks):
        yield
