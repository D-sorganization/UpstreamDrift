"""Local conftest: ensure real PyQt6 is loaded for the chat_history UI tests.

The repo-root conftest replaces ``PyQt6`` with a ``MagicMock`` early so
that test collection survives on machines where PyQt6 binaries are
unhealthy. The chat_history launcher tests require a working ``QWidget``
hierarchy (real ``QPlainTextEdit``, ``QLineEdit``, etc.) so this conftest
restores the real package by clearing the mocked entries from
``sys.modules`` and importing PyQt6 fresh. If the real binary cannot
load, every test in this directory is skipped.
"""

from __future__ import annotations

import os
import sys
import unittest.mock

import pytest

# Prefer offscreen platform so Qt does not try to open a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_PYQT_MODULES = [
    name for name in list(sys.modules) if name == "PyQt6" or name.startswith("PyQt6.")
]
_replaced = False
for name in _PYQT_MODULES:
    mod = sys.modules.get(name)
    if isinstance(mod, unittest.mock.MagicMock):
        sys.modules.pop(name, None)
        _replaced = True

try:
    import PyQt6  # noqa: F401  # re-imports the real package
    import PyQt6.QtCore  # noqa: F401
    import PyQt6.QtGui  # noqa: F401
    import PyQt6.QtWidgets  # noqa: F401
except Exception:  # noqa: BLE001
    pytest.skip(
        "Real PyQt6 is not available in this environment; chat_history "
        "launcher UI tests require a working PyQt6 install.",
        allow_module_level=True,
    )


@pytest.fixture(scope="session", autouse=True)
def _qapp_singleton():
    """Provide a single QApplication for every test in this directory."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
