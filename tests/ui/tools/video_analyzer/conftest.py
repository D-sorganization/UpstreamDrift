"""Fixtures for the Video Analyzer GUI tests.

Forces the offscreen Qt platform before PyQt6 import so the widgets
construct on headless CI runners, and skips the whole package cleanly
when PyQt6 is unavailable. Mirrors
``tests/ui/tools/simulation_backends/conftest.py``.
"""

from __future__ import annotations

import os

import pytest

# Headless rendering must be configured before PyQt6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Skip the entire package cleanly when PyQt6 is not installed.
PyQt6 = pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped ``QApplication`` singleton for the widget tests."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
