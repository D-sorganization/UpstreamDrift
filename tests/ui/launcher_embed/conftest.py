"""Local fixtures for the embedded-host UI tests.

Mirrors the lightweight ``qapp`` fixture pattern used elsewhere in the
launcher test suite (see ``tests/launchers/conftest.py``). Forces the
offscreen Qt platform so headless CI runners can construct widgets, and
clears the embeddable-tool registry between tests to avoid leaking
fixture tools across cases.
"""

from __future__ import annotations

import os

import pytest

# Force Qt to use the offscreen platform so headless CI / sandboxed
# runners do not fail to instantiate widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped ``QApplication`` instance."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _clear_embed_registry():
    """Clear the embeddable-tool registry between tests."""
    from src.shared.python.launcher_embed import (
        EMBEDDABLE_TOOL_REGISTRY,
    )

    EMBEDDABLE_TOOL_REGISTRY.clear()
    yield
    EMBEDDABLE_TOOL_REGISTRY.clear()
