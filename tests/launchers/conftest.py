"""Shared fixtures for launcher tests.

Provides a lightweight ``qapp`` fixture that mirrors the one supplied by
``pytest-qt``.  ``pytest-qt`` is intentionally not in the project's
declared dependencies (see issue #4676), so the launcher test suite
needs its own fallback to construct a single ``QApplication`` for the
entire test session.

Also hosts the shot-tracer ``mock_flight_models``/``tracer_widget``
fixtures (moved here from ``test_shot_tracer.py``, ADR-0047 H2, #9351):
conftest fixtures are auto-discovered by every test module under this
directory without an import, avoiding the ruff ``F811`` "redefinition"
false positive that importing a fixture function into another test
module triggers.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import MagicMock, patch

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


@pytest.fixture
def mock_flight_models() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    class MockModelType:
        value = "mock"

    class MockModel:
        name = "Mock Model"
        description = "Mock Description"
        reference = "Mock Ref"

    with (
        patch(
            "src.launchers.shot_tracer.FlightModelType", [MockModelType]
        ) as ModelTypeMock,
        patch("src.launchers.shot_tracer.FlightModelRegistry") as RegistryMock,
    ):
        RegistryMock.get_model.return_value = MockModel()
        yield ModelTypeMock, RegistryMock


@pytest.fixture
def tracer_widget(qapp, mock_flight_models) -> Generator[object, None, None]:
    from PyQt6.QtWidgets import QWidget

    from src.launchers.shot_tracer import MultiModelShotTracerWidget

    with patch("src.launchers.shot_tracer.PYQTGRAPH_AVAILABLE", False):
        parent_widget = QWidget()
        widget = MultiModelShotTracerWidget(parent=parent_widget)
        yield widget
