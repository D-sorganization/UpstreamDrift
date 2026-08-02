"""Conftest for plotting tests — pre-mocks PyQt6 to avoid DLL crashes."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock


import pytest


@pytest.fixture(autouse=True, scope="session")
def _mock_pyqt6():
    """Inject mock PyQt6 modules so plotting imports don't crash.

    This is only needed because PyQt6 DLLs may be broken in headless
    CI or when the Qt binaries are incomplete.  The plotting renderers
    themselves only use matplotlib (not Qt).
    """
    mocked_modules = [
        "PyQt6",
        "PyQt6.QtWidgets",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "src.shared.python.ui",
        "src.shared.python.ui.qt",
        "src.shared.python.ui.qt.plotting",
        "src.shared.python.ui.loading_button",
    ]
    saved = {}
    for mod_name in mocked_modules:
        if mod_name in sys.modules:
            saved[mod_name] = sys.modules[mod_name]
        sys.modules[mod_name] = MagicMock(spec=ModuleType)

    yield

    for mod_name in mocked_modules:
        if mod_name in saved:
            sys.modules[mod_name] = saved[mod_name]
        else:
            sys.modules.pop(mod_name, None)
