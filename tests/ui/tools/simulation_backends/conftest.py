"""Fixtures for the Simulation Backends launcher UI tests.

Forces the offscreen Qt platform and the Agg matplotlib backend so the
widgets construct on headless CI runners, skips the whole package when
PyQt6 is unavailable, and clears the embeddable-tool registry between
tests so the adapter's self-registration does not leak across cases.
"""

from __future__ import annotations

import os

import pytest

# Headless rendering must be configured before PyQt6 / matplotlib import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

# Skip the entire package cleanly when PyQt6 is not installed.
PyQt6 = pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped ``QApplication`` singleton for the widget tests."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _clear_embed_registry():
    """Clear the embeddable-tool registry before and after each test."""
    from src.shared.python.launcher_embed import EMBEDDABLE_TOOL_REGISTRY

    EMBEDDABLE_TOOL_REGISTRY.clear()
    yield
    EMBEDDABLE_TOOL_REGISTRY.clear()
