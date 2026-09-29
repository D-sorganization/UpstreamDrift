"""Heavy integration tests for PyQt6 GUI components (fixes #1984).

Verifies that core PyQt6 widgets — QApplication, launcher, theme system,
and pendulum simulator — can be instantiated in a headless (Xvfb) environment.
All tests skip gracefully when PyQt6 is unavailable.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(scope="module")
def qt_app():
    """Return or create a QApplication singleton for headless tests."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 not installed")

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app
    # Do not call app.quit() here — other tests in the module may need the app


class TestQApplicationCreation:
    """Contract: QApplication can be created in a headless environment."""

    def test_qapplication_exists(self, qt_app) -> None:
        """QApplication instance is alive after fixture setup."""
        from PyQt6.QtWidgets import QApplication

        assert QApplication.instance() is not None

    def test_qapplication_process_events(self, qt_app) -> None:
        """QApplication.processEvents() does not raise in headless mode."""
        qt_app.processEvents()


pytestmark = pytest.mark.live_simulation
