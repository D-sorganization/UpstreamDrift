"""Local fixtures for launcher-layout tests.

The repo's root conftest mocks PyQt6 when the binding is not yet
loaded (Windows CI without PyQt6 installed). Qt-dependent tests can
opt in to a real-Qt check via :data:`qt_real` and skip when running
against the DummyWidget mock.
"""

from __future__ import annotations

import sys

import pytest


def _pyqt6_is_real() -> bool:
    """Return True only if PyQt6 is the real binding (not the conftest mock)."""
    if "PyQt6.QtWidgets" not in sys.modules:
        return False
    mod = sys.modules["PyQt6.QtWidgets"]
    return (
        getattr(mod, "__name__", "") == "PyQt6.QtWidgets"
        and type(mod).__name__ == "module"
        and not type(mod).__module__.startswith("unittest.mock")
    )


@pytest.fixture
def qt_real() -> None:
    """Skip the test when PyQt6 is mocked by the root conftest."""
    if not _pyqt6_is_real():
        pytest.skip("PyQt6 mocked by conftest — Qt-dependent test skipped")
