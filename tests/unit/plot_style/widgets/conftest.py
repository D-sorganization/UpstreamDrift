"""Shared fixtures for plot-style widget tests."""

from __future__ import annotations

import os

import pytest

# Force the offscreen Qt platform so the suite is headless-safe even on
# systems without a display server (CI, containers, ssh).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Pin pytest-qt to the PyQt6 backend that the production widgets target;
# without this, pytest-qt may auto-select PySide6 and instantiate widgets
# of the wrong type, which then fail ``isinstance`` checks in qtbot.
os.environ["PYTEST_QT_API"] = "pyqt6"


@pytest.fixture(autouse=True)
def _ensure_qapp(qapp):  # type: ignore[no-untyped-def]
    """Ensure ``QApplication`` exists for every widget test."""
    return qapp
