"""Tests for src.launchers.app_zoom — config + install delegation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.launchers import app_zoom


class TestZoomConfig:
    def test_config_values(self) -> None:
        cfg = app_zoom.UPSTREAM_DRIFT_ZOOM_CONFIG
        assert cfg.minimum_percent == 60
        assert cfg.maximum_percent == 180
        assert cfg.default_percent == 100
        assert cfg.step_percent == 10
        assert cfg.settings_key == "ui_zoom_percent"
        assert cfg.settings_app == "UpstreamDrift"

    def test_default_within_bounds(self) -> None:
        cfg = app_zoom.UPSTREAM_DRIFT_ZOOM_CONFIG
        assert cfg.minimum_percent <= cfg.default_percent <= cfg.maximum_percent

    def test_step_positive(self) -> None:
        assert app_zoom.UPSTREAM_DRIFT_ZOOM_CONFIG.step_percent > 0


class TestInstallGlobalUiZoom:
    def test_delegates_to_install_application_zoom(self) -> None:
        fake_app = MagicMock()
        fake_settings = MagicMock()
        fake_controller = MagicMock()

        with patch.object(
            app_zoom, "install_application_zoom", return_value=fake_controller
        ) as m:
            result = app_zoom.install_global_ui_zoom(fake_app, fake_settings)
            assert result is fake_controller
            m.assert_called_once_with(
                fake_app, app_zoom.UPSTREAM_DRIFT_ZOOM_CONFIG, fake_settings
            )

    def test_default_settings_is_none(self) -> None:
        fake_app = MagicMock()
        with patch.object(
            app_zoom, "install_application_zoom", return_value=MagicMock()
        ) as m:
            app_zoom.install_global_ui_zoom(fake_app)
            args = m.call_args.args
            assert args[0] is fake_app
            assert args[1] is app_zoom.UPSTREAM_DRIFT_ZOOM_CONFIG
            assert args[2] is None
