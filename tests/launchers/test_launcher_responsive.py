"""Tests for responsive sizing and zoom filtering in UpstreamDrift launcher."""

import inspect

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication


class _Angle:
    def __init__(self, y: int) -> None:
        self._y = y

    def y(self) -> int:
        return self._y


class _WheelEvent:
    def __init__(self, modifiers: object, delta_y: int) -> None:
        self._modifiers = modifiers
        self._delta_y = delta_y
        self.accepted = False

    def modifiers(self) -> object:
        return self._modifiers

    def angleDelta(self) -> _Angle:  # noqa: N802
        return _Angle(self._delta_y)

    def accept(self) -> None:
        self.accepted = True


class _Settings:
    def __init__(self) -> None:
        self.saved: dict[str, object] = {}

    def value(
        self,
        key: str,
        defaultValue: object | None = None,
        **_kwargs: object,
    ) -> object:
        return self.saved.get(key, defaultValue)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        self.saved[key] = value


def test_launcher_ui_setup_responsive(qapp: QApplication) -> None:
    """Launcher search and zoom controls avoid fixed-width clipping."""
    from src.launchers.launcher_ui_setup import LauncherUISetupMixin

    source = inspect.getsource(LauncherUISetupMixin._setup_top_bar_status_and_search)
    zoom_source = inspect.getsource(LauncherUISetupMixin._setup_view_mode_and_zoom)

    assert "set_text_minimum_width" in source
    assert "TextWidthSpec(minimum_px=250)" in source
    assert "self.zoom_slider.setMinimumWidth(140)" in zoom_source
    assert "self.zoom_slider.setFixedWidth(140)" not in zoom_source


def test_cross_engine_dashboard_responsive(qapp: QApplication) -> None:
    """Cross-engine dashboard config panel uses a responsive minimum width."""
    try:
        from src.launchers.cross_engine_dashboard import _create_dashboard_window_class
    except ImportError:
        pytest.skip("matplotlib not available in this environment, skipping cross_engine_dashboard test")

    WindowCls = _create_dashboard_window_class()
    source = inspect.getsource(WindowCls)

    assert "panel.setMinimumWidth(260)" in source
    assert "panel.setFixedWidth(260)" not in source


def test_application_zoom_controller_filtering(qapp: QApplication) -> None:
    """ApplicationZoomController intercepts Ctrl+Wheel but ignores plain wheel."""
    try:
        from src.shared.python.theme.zoom import ApplicationZoomController, ZoomConfig

        settings = _Settings()
        controller = ApplicationZoomController(
            qapp,
            ZoomConfig(minimum_percent=60, maximum_percent=180, settings_key="zoom"),
            settings,
        )
    except ImportError:
        pytest.skip("ApplicationZoomController not available in this environment")

    plain_event = _WheelEvent(Qt.KeyboardModifier.NoModifier, 120)
    assert controller._handle_wheel(plain_event) is False
    assert controller.zoom_percent == 100

    ctrl_event = _WheelEvent(Qt.KeyboardModifier.ControlModifier, 120)
    assert controller._handle_wheel(ctrl_event) is True
    assert ctrl_event.accepted is True
    assert controller.zoom_percent == 110
    assert settings.saved["zoom"] == 110
