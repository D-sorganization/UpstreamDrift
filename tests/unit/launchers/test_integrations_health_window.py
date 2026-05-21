"""Tests for the UD launcher's integrations health window wrapper.

The wrapper is intentionally thin: ``make_dashboard_widget`` returns an
instance of the shared dashboard widget (Tools PR #2914), and
``open_integrations_health_window`` mounts that widget in a modeless
QDialog. Most behaviour is exercised by the Tools-side widget tests.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

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

    def test_uses_injected_provider(self, qapp: QApplication, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.launchers.integrations_health_window._shared_dashboard_widget_class",
            lambda: None,
        )
        calls: list[int] = []

        def provider() -> list:
            calls.append(1)
            return []

        widget = make_dashboard_widget(status_provider=provider)
        widget.refresh()
        assert calls  # provider was called

    def test_falls_back_when_shared_widget_is_unavailable(
        self, qapp: QApplication, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "src.launchers.integrations_health_window._shared_dashboard_widget_class",
            lambda: None,
        )
        from src.launchers.integrations_health_panel import IntegrationsHealthPanel

        widget = make_dashboard_widget(status_provider=list)

        assert isinstance(widget, IntegrationsHealthPanel)


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

    def test_dialog_uses_fallback_panel_when_shared_widget_is_unavailable(
        self, qapp: QApplication, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "src.launchers.integrations_health_window._shared_dashboard_widget_class",
            lambda: None,
        )
        from src.launchers.integrations_health_panel import IntegrationsHealthPanel

        dialog = open_integrations_health_window()

        assert dialog.findChild(IntegrationsHealthPanel) is not None
        dialog.close()
