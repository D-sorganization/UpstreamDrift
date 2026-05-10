"""Local fixtures for the C3D Viewer embed-adapter tests.

Mirrors the offscreen-Qt + ``qapp`` pattern used by the launcher embed
UI tests. We don't auto-clear the embeddable-tool registry here: the
adapter under test registers itself at import time, and other adapters
in the suite (e.g. pose_subscriber_demo) may legitimately share the
process-wide registry.
"""

from __future__ import annotations

import os

import pytest

# Force Qt to use the offscreen platform so headless CI / sandboxed
# runners can construct widgets without an X server.
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
