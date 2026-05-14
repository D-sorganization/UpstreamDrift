"""Application zoom wiring for the UpstreamDrift desktop launcher."""

from __future__ import annotations

from PyQt6.QtWidgets import QApplication

from src.shared.python.theme.zoom import (
    ApplicationZoomController,
    ZoomConfig,
    ZoomSettings,
    install_application_zoom,
)

UPSTREAM_DRIFT_ZOOM_CONFIG = ZoomConfig(
    minimum_percent=60,
    maximum_percent=180,
    default_percent=100,
    step_percent=10,
    settings_key="ui_zoom_percent",
    settings_app="UpstreamDrift",
)


def install_global_ui_zoom(
    app: QApplication,
    settings: ZoomSettings | None = None,
) -> ApplicationZoomController:
    """Install app-wide UI zoom without coupling to tile or canvas zoom."""
    return install_application_zoom(app, UPSTREAM_DRIFT_ZOOM_CONFIG, settings)
