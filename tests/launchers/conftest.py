"""Shared fixtures for launcher tests.

Provides a lightweight ``qapp`` fixture that mirrors the one supplied by
``pytest-qt``.  ``pytest-qt`` is intentionally not in the project's
declared dependencies (see issue #4676), so the launcher test suite
needs its own fallback to construct a single ``QApplication`` for the
entire test session.
"""

from __future__ import annotations

import os

import pytest

# Force Qt to use the offscreen platform so headless CI / sandboxed
# runners do not fail to instantiate widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped ``QApplication`` instance.

    Uses ``QApplication.instance()`` first to avoid the ``QApplication``
    singleton conflict that arises if the application is constructed
    twice.
    """
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
