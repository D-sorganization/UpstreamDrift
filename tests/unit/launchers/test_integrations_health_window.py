"""Tests for the UD launcher's integrations health window wrapper.

The wrapper is intentionally thin: ``make_dashboard_widget`` returns an
instance of the shared dashboard widget (Tools PR #2914), and
``open_integrations_health_window`` mounts that widget in a modeless
QDialog. Most behaviour is exercised by the Tools-side widget tests.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6.QtWidgets")
# Until Tools PR #2914 merges and the vendor submodule is bumped, the
# shared widget package may not be importable in CI.
pytest.importorskip(
    "src.shared.python.ai.mcp.widgets",
    reason="Pending Tools PR #2914 / vendor bump",
)

from PyQt6.QtWidgets import QApplication, QDialog, QWidget  # noqa: E402

from src.launchers.integrations_health_window import (  # noqa: E402
    make_dashboard_widget,
    open_integrations_health_window,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.mark.unit
class TestMakeDashboardWidget:
    def test_returns_qwidget(self, qapp: QApplication) -> None:
        widget = make_dashboard_widget(status_provider=list)
        assert isinstance(widget, QWidget)

    def test_uses_injected_provider(self, qapp: QApplication) -> None:
        calls: list[int] = []

        def provider() -> list:
            calls.append(1)
            return []

        widget = make_dashboard_widget(status_provider=provider)
        widget.refresh()
        assert calls  # provider was called


@pytest.mark.unit
class TestOpenIntegrationsHealthWindow:
    def test_returns_qdialog(self, qapp: QApplication) -> None:
        dialog = open_integrations_health_window()
        assert isinstance(dialog, QDialog)
        # Defensive cleanup.
        dialog.close()

    def test_dialog_has_title(self, qapp: QApplication) -> None:
        dialog = open_integrations_health_window()
        assert "Integrations" in dialog.windowTitle()
        dialog.close()
