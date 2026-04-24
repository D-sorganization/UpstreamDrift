"""Conftest for plotting tests — pre-mocks PyQt6 to avoid DLL crashes."""

from __future__ import annotations

import contextlib
from importlib.util import find_spec
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

_PYQT6_MODULES = [
    "PyQt6",
    "PyQt6.QtWidgets",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "src.shared.python.ui",
    "src.shared.python.ui.qt",
    "src.shared.python.ui.qt.plotting",
    "src.shared.python.ui.loading_button",
]

# Only mock PyQt6 if it is not installed.  If the real package is available,
# the tests use it directly and no mock is needed.  When PyQt6 is absent,
# we install mocks at collection time using patch.dict so they are removed
# cleanly when pytest exits.
_pyqt6_mock_stack = contextlib.ExitStack()

if find_spec("PyQt6") is None:
    _mocks = {mod: MagicMock(spec=ModuleType) for mod in _PYQT6_MODULES}
    _pyqt6_mock_stack.enter_context(patch.dict("sys.modules", _mocks))


def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove PyQt6 mocks installed at collection time (no-op if not installed)."""
    _pyqt6_mock_stack.close()
