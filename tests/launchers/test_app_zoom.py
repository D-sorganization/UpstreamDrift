"""Tests for launcher-level application zoom wiring."""

from __future__ import annotations


def test_install_global_ui_zoom_uses_upstream_drift_settings(monkeypatch) -> None:
    from src.launchers import app_zoom

    calls: list[tuple[object, object, object]] = []
    expected_controller = object()

    def fake_install_application_zoom(app, config=None, settings=None):
        calls.append((app, config, settings))
        return expected_controller

    monkeypatch.setattr(
        app_zoom,
        "install_application_zoom",
        fake_install_application_zoom,
    )
    qt_app = object()
    settings = object()

    controller = app_zoom.install_global_ui_zoom(qt_app, settings=settings)

    assert controller is expected_controller
    assert calls == [(qt_app, app_zoom.UPSTREAM_DRIFT_ZOOM_CONFIG, settings)]
    assert app_zoom.UPSTREAM_DRIFT_ZOOM_CONFIG.settings_app == "UpstreamDrift"
    assert app_zoom.UPSTREAM_DRIFT_ZOOM_CONFIG.settings_key == "ui_zoom_percent"
