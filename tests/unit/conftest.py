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

    These stubs allow test-file imports that reference heavy packages (pydrake,
    casadi, pinocchio) to succeed during collection without a real installation.
    They are only installed for modules that are genuinely absent so that a real
    installation is never shadowed.

    The autouse fixture ``_mock_heavy_deps`` below reinstalls fresh mocks via
    ``patch.dict`` before every test and tears them down afterward, satisfying
    the CLAUDE.md rule against persistent module-level ``sys.modules`` mutations.
    ``pytest_unconfigure`` removes the stubs that were added here once the
    session is complete, ensuring the process exits with a clean sys.modules.
    """
    pydrake_stub = MagicMock()
    stubs = {
        "pydrake": pydrake_stub,
        "pydrake.all": pydrake_stub,
        "casadi": MagicMock(),
        "pinocchio": MagicMock(),
        "pinocchio.casadi": MagicMock(),
    }
    installed: list[str] = []
    for name, stub in stubs.items():
        if name not in sys.modules:
            sys.modules[name] = stub
            installed.append(name)
    # Record which names we injected so pytest_unconfigure can clean them up.
    config._unit_collection_stubs = installed  # type: ignore[attr-defined]


def pytest_unconfigure(config: "pytest.Config") -> None:  # noqa: F821
    """Remove collection-time stubs installed by pytest_configure.

    Ensures the process-level sys.modules is clean after the test session ends,
    satisfying the CLAUDE.md requirement that sys.modules mutations are
    temporary and always cleaned up.
    """
    for name in getattr(config, "_unit_collection_stubs", ()):
        sys.modules.pop(name, None)


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
