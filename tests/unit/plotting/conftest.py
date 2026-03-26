"""Conftest for plotting tests — pre-mocks PyQt6 to avoid DLL crashes."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock


def _mock_pyqt6() -> None:
    """Inject mock PyQt6 modules so plotting imports don't crash.

    This is only needed because PyQt6 DLLs may be broken in headless
    CI or when the Qt binaries are incomplete.  The plotting renderers
    themselves only use matplotlib (not Qt).
    """
    for mod_name in [
        "PyQt6",
        "PyQt6.QtWidgets",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "src.shared.python.ui",
        "src.shared.python.ui.qt",
        "src.shared.python.ui.qt.plotting",
        "src.shared.python.ui.loading_button",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock(spec=ModuleType)


_mock_pyqt6()
