"""Conftest for plotting tests — pre-mocks PyQt6 to avoid DLL crashes."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# Modules replaced with stand-ins for the duration of a single plotting test.
#
# Issue #9188: this used to be a ``scope="session"`` fixture.  A session-scoped
# fixture declared in a *directory* conftest is created lazily when the first
# test in that directory runs, but it is only finalized at the end of the whole
# session — so the stubs stayed in ``sys.modules`` for every test collected
# after ``tests/unit/plotting``.  Because ``MagicMock(spec=ModuleType)`` raises
# ``AttributeError`` for any name a real module would provide, every later
# ``from PyQt6.QtCore import Qt`` (or ``QApplication``/``QDialog``) failed with
# ``cannot import name 'Qt' from '<unknown module name>' (unknown location)``.
# Which tests got hit depended purely on collection order, which is what made
# ``unit-test-gate`` a lottery.
#
# The stubs are still installed for these tests, but now via ``monkeypatch``,
# whose teardown is automatic and exception-safe.
_MOCKED_MODULES = (
    "PyQt6",
    "PyQt6.QtWidgets",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "src.shared.python.ui",
    "src.shared.python.ui.qt",
    "src.shared.python.ui.qt.plotting",
    "src.shared.python.ui.loading_button",
)


@pytest.fixture(autouse=True)
def _mock_pyqt6(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject mock PyQt6 modules so plotting imports don't crash.

    This is only needed because PyQt6 DLLs may be broken in headless
    CI or when the Qt binaries are incomplete.  The plotting renderers
    themselves only use matplotlib (not Qt).

    ``monkeypatch.setitem`` restores the previous value on teardown — and
    deletes the key outright when there was no previous value — so these
    stubs can never outlive the test that asked for them.
    """
    for mod_name in _MOCKED_MODULES:
        monkeypatch.setitem(sys.modules, mod_name, MagicMock(spec=ModuleType))
