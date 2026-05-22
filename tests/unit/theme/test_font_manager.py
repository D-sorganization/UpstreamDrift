"""Tests for shared font manager helpers."""

from __future__ import annotations

import logging
from typing import Any

import pytest

import src.shared.python.theme.font_manager as font_module
from src.shared.python.theme.font_manager import BUILTIN_FONTS, FontManager


class _FakeSettings:
    store: dict[tuple[str, str], Any] = {}

    def __init__(self, organization: str, application: str) -> None:
        self.organization = organization
        self.application = application
        self.group = ""

    def beginGroup(self, group: str) -> None:  # noqa: N802
        self.group = group

    def value(self, key: str, default: Any = None, type: object | None = None) -> Any:
        value = self.store.get((self.group, key), default)
        if type is not None:
            return type(value)  # type: ignore[operator]
        return value

    def setValue(self, key: str, value: Any) -> None:  # noqa: N802
        self.store[(self.group, key)] = value


class _FakeFontDatabase:
    families_result: list[str] = []

    @staticmethod
    def families() -> list[str]:
        return list(_FakeFontDatabase.families_result)


class _FakeFont:
    def __init__(self, family: str | None = None) -> None:
        self.family = family
        self.point_size: int | None = None

    def setPointSize(self, point_size: int) -> None:  # noqa: N802
        self.point_size = point_size


class _FakeApplication:
    _instance: _FakeApplication | None = None

    def __init__(self) -> None:
        self.font: _FakeFont | None = None

    @classmethod
    def instance(cls) -> _FakeApplication | None:
        return cls._instance

    def setFont(self, font: _FakeFont) -> None:  # noqa: N802
        self.font = font


@pytest.fixture(autouse=True)
def _font_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeSettings.store = {}
    _FakeFontDatabase.families_result = []
    _FakeApplication._instance = None
    FontManager._instance = None
    monkeypatch.setattr(font_module, "QSettings", _FakeSettings)
    monkeypatch.setattr(font_module, "QFontDatabase", _FakeFontDatabase)
    monkeypatch.setattr(font_module, "QFont", _FakeFont)
    monkeypatch.setattr(font_module, "QApplication", _FakeApplication)


def test_font_manager_loads_default_font_and_settings_group() -> None:
    manager = FontManager(app_context="Launcher")

    assert manager.get_current_font() == "Inter"
    assert manager.settings.group == "Font_Launcher"


def test_font_manager_loads_saved_font_preference() -> None:
    _FakeSettings.store[("Font_Global", "font_family")] = "Roboto"

    manager = FontManager()

    assert manager.get_current_font() == "Roboto"


def test_get_available_fonts_filters_builtin_fonts_and_adds_system_default() -> None:
    _FakeFontDatabase.families_result = ["Roboto", "Arial", "Some Other Font"]

    manager = FontManager()

    assert manager.get_available_fonts() == ["Roboto", "Arial", "System Default"]


def test_get_available_fonts_keeps_system_default_when_no_builtin_fonts() -> None:
    _FakeFontDatabase.families_result = ["Unrelated"]

    manager = FontManager()

    assert manager.get_available_fonts() == ["System Default"]


def test_change_font_persists_applies_and_emits_for_new_font(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = FontManager()
    applied: list[str] = []
    emitted: list[str] = []
    monkeypatch.setattr(
        manager, "apply_font", lambda: applied.append(manager.current_font)
    )
    manager.fontChanged.connect(emitted.append)

    manager.change_font("Roboto")

    assert manager.get_current_font() == "Roboto"
    assert _FakeSettings.store[("Font_Global", "font_family")] == "Roboto"
    assert applied == ["Roboto"]
    assert emitted == ["Roboto"]


def test_change_font_noops_when_font_is_already_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = FontManager()
    monkeypatch.setattr(manager, "apply_font", lambda: pytest.fail("should not apply"))

    manager.change_font("Inter")

    assert "font_family" not in {key for _group, key in _FakeSettings.store}


def test_apply_font_uses_current_font_with_standard_point_size() -> None:
    app = _FakeApplication()
    manager = FontManager()
    manager.change_font("Roboto")

    manager.apply_font(app)

    assert app.font is not None
    assert app.font.family == "Roboto"
    assert app.font.point_size == 10


def test_apply_font_uses_qt_default_for_system_default() -> None:
    app = _FakeApplication()
    manager = FontManager()
    manager.current_font = "System Default"

    manager.apply_font(app)

    assert app.font is not None
    assert app.font.family is None


def test_apply_font_uses_application_singleton() -> None:
    app = _FakeApplication()
    _FakeApplication._instance = app
    manager = FontManager()

    manager.apply_font()

    assert app.font is not None
    assert app.font.family == "Inter"


def test_apply_font_logs_warning_without_application(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = FontManager()

    with caplog.at_level(logging.WARNING):
        manager.apply_font()

    assert "No QApplication instance" in caplog.text


def test_get_font_manager_reuses_singleton() -> None:
    first = font_module.get_font_manager(app_context="Launcher")
    second = font_module.get_font_manager(app_context="Other")

    assert first is second
    assert second.app_context == "Launcher"
    assert BUILTIN_FONTS[0] == "Inter"
