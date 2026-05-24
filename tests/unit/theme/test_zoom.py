"""Tests for shared application zoom helpers."""

from __future__ import annotations

import pytest

from src.shared.python.theme.zoom import (
    ApplicationZoomController,
    ZoomConfig,
    ZoomTokenSet,
    install_application_zoom,
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


class _KeyEvent:
    def __init__(self, modifiers: object, key: object) -> None:
        self._modifiers = modifiers
        self._key = key
        self.accepted = False

    def modifiers(self) -> object:
        return self._modifiers

    def key(self) -> object:
        return self._key

    def accept(self) -> None:
        self.accepted = True


def test_zoom_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="minimum_percent"):
        ZoomConfig(minimum_percent=150, maximum_percent=100)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minimum_percent": 0}, "minimum_percent"),
        ({"maximum_percent": 0}, "maximum_percent"),
        ({"step_percent": 0}, "step_percent"),
        ({"minimum_percent": 80, "default_percent": 60}, "default_percent"),
        ({"maximum_percent": 120, "default_percent": 140}, "default_percent"),
    ],
)
def test_zoom_config_rejects_invalid_positive_and_default_contracts(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ZoomConfig(**kwargs)


def test_scale_px_uses_percent_with_positive_contract() -> None:
    assert scale_px(80, 125) == 100
    assert scale_px(1, 50) == 1

    with pytest.raises(ValueError, match="zoom_percent"):
        scale_px(80, 0)

    with pytest.raises(ValueError, match="value"):
        scale_px(0, 100)


def test_zoom_tokens_scale_from_percent() -> None:
    tokens = ZoomTokenSet.from_percent(150)

    assert tokens.font_px == 18
    assert tokens.label_font_px == 16
    assert tokens.padding_px == 12
    assert tokens.spacing_px == 9
    assert tokens.icon_px == 24
    assert tokens.minimum_control_px == 120


@pytest.mark.parametrize(
    ("stored_value", "expected"),
    [
        ("125", 125),
        (250, 180),
        (None, 90),
        (object(), 90),
    ],
)
def test_controller_loads_persisted_zoom_with_default_and_clamping(
    qapp,
    stored_value: object,
    expected: int,
) -> None:
    controller = ApplicationZoomController(
        qapp,
        ZoomConfig(
            minimum_percent=60,
            maximum_percent=180,
            default_percent=90,
            settings_key="zoom",
        ),
        _Settings(stored_value),
    )

    assert controller.zoom_percent == expected


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


def test_controller_does_not_persist_or_emit_when_zoom_is_unchanged(qapp) -> None:
    settings = _Settings(100)
    controller = ApplicationZoomController(
        qapp, ZoomConfig(settings_key="zoom"), settings
    )
    emitted: list[int] = []
    controller.zoomChanged.connect(emitted.append)

    controller.set_zoom_percent(100)

    assert settings.saved == {}
    assert emitted == []


def test_controller_clamps_persists_and_emits_zoom_changes(qapp) -> None:
    settings = _Settings(100)
    controller = ApplicationZoomController(
        qapp,
        ZoomConfig(minimum_percent=80, maximum_percent=120, settings_key="zoom"),
        settings,
    )
    emitted: list[int] = []
    controller.zoomChanged.connect(emitted.append)

    controller.set_zoom_percent(999)

    assert controller.zoom_percent == 120
    assert settings.saved == {"zoom": 120}
    assert emitted == [120]


def test_controller_keyboard_shortcuts_handle_zoom_and_reset(qapp) -> None:
    from PyQt6.QtCore import Qt

    settings = _Settings(100)
    controller = ApplicationZoomController(
        qapp,
        ZoomConfig(default_percent=100, step_percent=25, settings_key="zoom"),
        settings,
    )

    plus = _KeyEvent(Qt.KeyboardModifier.ControlModifier, Qt.Key.Key_Equal)
    assert controller._handle_key(plus) is True
    assert plus.accepted is True
    assert controller.zoom_percent == 125

    minus = _KeyEvent(Qt.KeyboardModifier.ControlModifier, Qt.Key.Key_Minus)
    assert controller._handle_key(minus) is True
    assert controller.zoom_percent == 100

    zero = _KeyEvent(Qt.KeyboardModifier.ControlModifier, Qt.Key.Key_0)
    controller.set_zoom_percent(150)
    assert controller._handle_key(zero) is True
    assert controller.zoom_percent == 100


def test_controller_ignores_non_control_and_unknown_keys(qapp) -> None:
    from PyQt6.QtCore import Qt

    controller = ApplicationZoomController(qapp, settings=_Settings(100))

    plain_plus = _KeyEvent(Qt.KeyboardModifier.NoModifier, Qt.Key.Key_Plus)
    assert controller._handle_key(plain_plus) is False
    assert plain_plus.accepted is False

    unknown = _KeyEvent(Qt.KeyboardModifier.ControlModifier, Qt.Key.Key_A)
    assert controller._handle_key(unknown) is False
    assert unknown.accepted is False
    assert controller.zoom_percent == 100


def test_install_application_zoom_installs_and_returns_controller(
    qapp, monkeypatch
) -> None:
    installed: list[object] = []
    monkeypatch.setattr(qapp, "installEventFilter", installed.append)

    controller = install_application_zoom(qapp, settings=_Settings(100))

    assert isinstance(controller, ApplicationZoomController)
    assert installed == [controller]
