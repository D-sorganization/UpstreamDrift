"""Test fixtures for the Option 4 Python ↔ Simscape bridge."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Make the option4 package importable as a top-level module so tests don't
# have to spell out the seven-deep package path.
_HERE = Path(__file__).resolve().parent
_PKG_DIR = _HERE.parent  # option4_python_bridge/

if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))


def _matlab_engine_importable() -> bool:
    try:
        return importlib.util.find_spec("matlab.engine") is not None
    except ModuleNotFoundError:
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``requires_matlab_engine`` tests when matlab.engine isn't installed.

    The skip is loud — pytest reports "skipped: matlab.engine not importable"
    so it is obvious why the round-trip and recovery tests didn't run.
    """
    if _matlab_engine_importable():
        return
    skip = pytest.mark.skip(
        reason=(
            "matlab.engine not importable — install via "
            "`python -m pip install matlabengine` (Python 3.9–3.12 only). "
            "See option4_python_bridge/INSTALLATION.md."
        )
    )
    for item in items:
        keywords = item.keywords
        if "requires_matlab_engine" in keywords or "requires_matlab" in keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def adapter():
    """Session-scoped SimscapeAdapter. Started once, closed at session end."""
    pytest.importorskip("matlab.engine", reason="matlab.engine not installed")
    from simscape_adapter import SimscapeAdapter

    a = SimscapeAdapter()
    a.start()
    try:
        yield a
    finally:
        a.close()


@pytest.fixture(scope="session")
def n_joints(adapter):
    # PolynomialInputValues.mat may be absent in some checkouts; fall back to
    # the canonical 28-joint config the rest of the suite assumes.
    return adapter.get_n_joints(default=28)


@pytest.fixture(scope="session")
def bounds(adapter, n_joints):
    return adapter.get_polynomial_bounds(n_joints)


@pytest.fixture(scope="session")
def has_simscape_multibody(adapter) -> bool:
    """True iff the running MATLAB has Simscape Multibody licensed.

    Tests that need a real forward sim should ``pytest.skip(...)`` when this
    is False, with a clear message.
    """
    try:
        result = adapter.engine.eval("license('test', 'Simscape_Multibody')", nargout=1)
        return bool(result)
    except Exception:  # noqa: BLE001
        return False
