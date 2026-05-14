"""Tests for shared application zoom helpers."""

from __future__ import annotations

import pytest

from src.shared.python.theme.zoom import (
    ApplicationZoomController,
    ZoomConfig,
    scale_px,
)


class _Settings:
    def __init__(self, value: object = 100) -> None:
        self.value_to_return = value
        self.saved: dict[str, object] = {}

    def value(
        self,
        key: str,
        defaultValue: object | None = None,
        **_kwargs: object,
    ) -> object:
        return (
            self.value_to_return if self.value_to_return is not None else defaultValue
        )

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        self.saved[key] = value


class _Angle:
    def __init__(self, y_value: int) -> None:
        self._y_value = y_value

    def y(self) -> int:
        return self._y_value


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


def test_zoom_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="minimum_percent"):
        ZoomConfig(minimum_percent=150, maximum_percent=100)


def test_scale_px_uses_percent_with_positive_contract() -> None:
    assert scale_px(80, 125) == 100

    with pytest.raises(ValueError, match="zoom_percent"):
        scale_px(80, 0)


def test_controller_handles_ctrl_wheel_without_touching_plain_wheel(qapp) -> None:
    from PyQt6.QtCore import Qt

    settings = _Settings()
    controller = ApplicationZoomController(
        qapp,
        ZoomConfig(minimum_percent=60, maximum_percent=180, settings_key="zoom"),
        settings,
    )

    plain_event = _WheelEvent(Qt.KeyboardModifier.NoModifier, 120)
    assert controller._handle_wheel(plain_event) is False
    assert controller.zoom_percent == 100

    ctrl_event = _WheelEvent(Qt.KeyboardModifier.ControlModifier, 120)
    assert controller._handle_wheel(ctrl_event) is True
    assert ctrl_event.accepted is True
    assert controller.zoom_percent == 110
    assert settings.saved["zoom"] == 110
