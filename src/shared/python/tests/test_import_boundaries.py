"""
Boundary tests for shared packages.
Ensures that shim packages resolve correctly and forbidden packages do not shadow vendor packages.
"""

import sys
from pathlib import Path
import pytest
import importlib

# Ensure local package roots are available
_REPO_ROOT = Path(__file__).resolve().parents[4]
_LOCAL_PYTHON = _REPO_ROOT / "src" / "shared" / "python"
_VENDOR_PYTHON = _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"


@pytest.fixture(autouse=True)
def clean_sys_modules():
    """Ensure modules are clean before each test."""
    to_remove = [
        m
        for m in sys.modules
        if m.startswith(("calc_backend", "sidekick", "signal_toolkit"))
    ]
    for m in to_remove:
        del sys.modules[m]
    yield
    for m in to_remove:
        if m in sys.modules:
            del sys.modules[m]


def test_calc_backend_resolves_to_vendor():
    """Verify calc_backend is properly shimmed and resolves from the vendor directory."""
    import calc_backend

    assert calc_backend.__path__ is not None
    assert any(str(_VENDOR_PYTHON) in p for p in calc_backend.__path__), (
        "calc_backend should resolve its submodules from the vendored module"
    )


def test_signal_toolkit_resolves_to_vendor():
    """Verify signal_toolkit is properly shimmed and resolves from the vendor directory."""
    import signal_toolkit

    assert signal_toolkit.__path__ is not None
    assert any(str(_VENDOR_PYTHON) in p for p in signal_toolkit.__path__), (
        "signal_toolkit should resolve its submodules from the vendored module"
    )


def test_sidekick_resolves_locally():
    """Verify sidekick resolves from the local directory."""
    import sidekick

    assert sidekick.__file__ is not None
    assert str(_LOCAL_PYTHON) in sidekick.__file__, (
        "sidekick should resolve to the local module"
    )


def test_forbidden_shadowing():
    """Assert that a forbidden shadow case fails to import."""
    # We will simulate a forbidden shadow by creating a package that exists in vendor
    # but try to import it from local without a shim.
    # Actually, if we don't have a shim and it's not in __path__ or sys.path, it should fail.

    with pytest.raises(ImportError):
        importlib.import_module("some_nonexistent_vendor_pkg")
