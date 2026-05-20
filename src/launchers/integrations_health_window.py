"""Launcher embed: Integrations Health dashboard window.

Thin wrapper that hosts the shared
:class:`IntegrationsHealthDashboardWidget` (from Tools PR #2914) inside
a top-level :class:`QDialog` so it can be opened from the launcher's
Window menu.

Architectural rule: the dashboard widget itself lives in Tools shared
because Gasification_Model will surface the same dashboard in its own
window. This module is the UpstreamDrift-specific "open this in a
window" affordance.

Implements UD #5643 (consumer side; widget lives in Tools).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QDialog, QWidget


def _shared_dashboard_widget_class() -> Any | None:
    """Return the shared Tools dashboard widget class when available."""
    try:
        from src.shared.python.ai.mcp.widgets import IntegrationsHealthDashboardWidget
    except ImportError:
        return None
    return IntegrationsHealthDashboardWidget


def _make_fallback_dashboard_widget(
    *, status_provider: Any | None = None, auto_refresh: bool = False
) -> Any:
    """Build the launcher-owned fallback panel for stale/missing Tools widgets."""
    from src.launchers.integrations_health_panel import IntegrationsHealthPanel

    return IntegrationsHealthPanel(
        status_provider=status_provider,
        auto_refresh=auto_refresh,
    )


def open_integrations_health_window(parent: QWidget | None = None) -> QDialog:
    """Open the integrations health dashboard as a modeless dialog.

    Args:
        parent: Standard Qt parent.

    Returns:
        The opened :class:`QDialog`. Modeless — caller can keep a
        reference to drive ``raise_()`` later.
    """
    from PyQt6.QtWidgets import QDialog, QVBoxLayout

    dialog = QDialog(parent)
    dialog.setWindowTitle("Integrations Health")
    dialog.resize(720, 360)

    dashboard_class = _shared_dashboard_widget_class()
    if dashboard_class is None:
        widget = _make_fallback_dashboard_widget()
        widget.setParent(dialog)
    else:
        widget = dashboard_class(parent=dialog)
    layout = QVBoxLayout(dialog)
    layout.addWidget(widget)
    widget.refresh()  # populate initial rows

    dialog.show()
    return dialog


def make_dashboard_widget(
    *, status_provider: Any | None = None, auto_refresh: bool = False
) -> Any:
    """Return a fresh :class:`IntegrationsHealthDashboardWidget`.

    Convenience factory for embedding the dashboard inside an existing
    container (instead of a standalone window). Kept as a free function
    so consumers don't need to import the shared widget directly.

    Args:
        status_provider: Optional zero-arg callable returning a list of
            :class:`IntegrationStatus`. Defaults to the canonical
            ``list_all_integrations``.
        auto_refresh: Enable the 30 s auto-refresh timer.

    Returns:
        Constructed :class:`IntegrationsHealthDashboardWidget`.
    """
    dashboard_class = _shared_dashboard_widget_class()
    if dashboard_class is None:
        return _make_fallback_dashboard_widget(
            status_provider=status_provider,
            auto_refresh=auto_refresh,
        )
    return dashboard_class(status_provider=status_provider, auto_refresh=auto_refresh)


__all__ = ["make_dashboard_widget", "open_integrations_health_window"]
