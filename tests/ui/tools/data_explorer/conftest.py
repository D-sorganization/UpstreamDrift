"""Local fixtures for the Data Explorer UI tests.

Mirrors the ``qapp`` fixture pattern used in
``tests/launchers/conftest.py`` so the Data Explorer tests can share a
session ``QApplication`` under ``QT_QPA_PLATFORM=offscreen``. Defined
locally because pytest's conftest discovery is ancestor-based and
``tests/launchers`` is a sibling, not an ancestor, of this directory.
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
