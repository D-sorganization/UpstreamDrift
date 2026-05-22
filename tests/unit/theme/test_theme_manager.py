"""Focused unit tests for ThemeManager persistence and config behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.theme import theme_manager as theme_module
from src.shared.python.theme.colors import THEME_COLOR_KEYS
from src.shared.python.theme.theme_manager import ThemeManager, get_theme_manager


VALID_COLORS: dict[str, str] = dict.fromkeys(THEME_COLOR_KEYS, "#aabbcc")


class FakeSettings:
    """Small QSettings stand-in that preserves constructor contract."""

    instances: list[FakeSettings] = []

    def __init__(self, org: str, app: str) -> None:
        self.org = org
        self.app = app
        self.values: dict[str, object] = {}
        FakeSettings.instances.append(self)

    def value(self, key: str, default: object = None, *, type: type | None = None):
        value = self.values.get(key, default)
        if type is not None and value is not None:
            return type(value)
        return value

    def setValue(self, key: str, value: object) -> None:
        self.values[key] = value


@pytest.fixture(autouse=True)
def reset_theme_manager() -> None:
    ThemeManager.reset_instance()
    FakeSettings.instances.clear()
    yield
    ThemeManager.reset_instance()
    FakeSettings.instances.clear()


@pytest.fixture()
def fake_settings(monkeypatch: pytest.MonkeyPatch) -> type[FakeSettings]:
    monkeypatch.setattr(theme_module, "QSettings", FakeSettings)
    return FakeSettings


@pytest.fixture()
def temp_theme_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_settings: type[FakeSettings]
) -> Path:
    theme_path = tmp_path / "user_themes.json"
    monkeypatch.setattr(
        ThemeManager,
        "_get_custom_theme_path",
        lambda self: theme_path,
    )
    return theme_path


def test_singleton_uses_first_settings_scope(fake_settings: type[FakeSettings]) -> None:
    first = get_theme_manager(settings_org="OrgA", settings_app="AppA")
    second = get_theme_manager(settings_org="OrgB", settings_app="AppB")

    assert first is second
    assert first.settings.org == "OrgA"
    assert first.settings.app == "AppA"
    assert len(fake_settings.instances) == 1


def test_context_inherits_global_theme_when_preference_is_inherit(
    temp_theme_path: Path,
) -> None:
    manager = ThemeManager(app_context="Launcher")
    manager.settings.values["theme"] = "Dark"
    manager.settings.values["theme_Launcher"] = "Inherit"

    assert manager._resolve_effective_theme() == "Dark"
    assert manager.get_theme_preference() == "Inherit"
    assert manager.get_available_themes()[0] == "Inherit"


def test_context_theme_preference_overrides_global_theme(temp_theme_path: Path) -> None:
    manager = ThemeManager(app_context="Launcher")
    manager.settings.values["theme"] = "Dark"
    manager.settings.values["theme_Launcher"] = "Light"

    assert manager._resolve_effective_theme() == "Light"
    assert manager.get_theme_preference() == "Light"


def test_change_theme_persists_global_preference_and_emits_signal(
    temp_theme_path: Path,
) -> None:
    manager = ThemeManager()
    emitted: list[str] = []
    manager.themeChanged.connect(emitted.append)

    manager.change_theme("Dark")

    assert manager.settings.values["theme"] == "Dark"
    assert manager.get_current_theme_name() == "Dark"
    assert emitted == ["Dark"]


def test_change_theme_to_inherit_persists_context_preference(
    temp_theme_path: Path,
) -> None:
    manager = ThemeManager(app_context="Launcher")
    manager.settings.values["theme"] = "Dark"

    manager.change_theme("Inherit")

    assert manager.settings.values["theme_Launcher"] == "Inherit"
    assert manager.get_current_theme_name() == "Dark"


def test_load_custom_themes_missing_file_returns_empty(temp_theme_path: Path) -> None:
    manager = ThemeManager()

    assert manager.custom_themes == {}
    assert manager._load_custom_themes() == {}


def test_load_custom_themes_discards_malformed_json(
    temp_theme_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    temp_theme_path.write_text("{not json", encoding="utf-8")

    manager = ThemeManager()

    assert manager.custom_themes == {}
    assert "Failed to load custom themes" in caplog.text


def test_load_custom_themes_filters_invalid_entries(temp_theme_path: Path) -> None:
    payload = {
        "Valid": {**VALID_COLORS, "unknown_key": "#ffffff"},
        "  Trimmed  ": VALID_COLORS,
        "": VALID_COLORS,
        "MissingKeys": {"bg": "#000000"},
        "BadColor": {**VALID_COLORS, "bg": "not-a-color"},
        "NotMapping": ["#ffffff"],
    }
    temp_theme_path.write_text(json.dumps(payload), encoding="utf-8")

    manager = ThemeManager()

    assert manager.custom_themes == {"Trimmed": VALID_COLORS, "Valid": VALID_COLORS}


def test_save_custom_theme_persists_normalised_colors_only(
    temp_theme_path: Path,
) -> None:
    manager = ThemeManager()
    colors = {**VALID_COLORS, "bg": "ABCDEF", "unknown_key": "#000000"}

    saved_name = manager.save_custom_theme("  Brand Theme  ", colors)

    assert saved_name == "Brand Theme"
    assert manager.custom_themes["Brand Theme"]["bg"] == "#abcdef"
    payload = json.loads(temp_theme_path.read_text(encoding="utf-8"))
    assert payload["Brand Theme"]["bg"] == "#abcdef"
    assert "unknown_key" not in payload["Brand Theme"]


def test_persist_custom_themes_logs_os_errors(
    monkeypatch: pytest.MonkeyPatch,
    temp_theme_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = ThemeManager()
    manager.custom_themes["Valid"] = VALID_COLORS

    def raise_os_error(self: ThemeManager) -> Path:
        raise OSError("read-only config")

    monkeypatch.setattr(ThemeManager, "_get_custom_theme_path", raise_os_error)

    manager._persist_custom_themes()

    assert "Failed to persist custom themes: read-only config" in caplog.text


def test_delete_active_custom_theme_falls_back_to_light(temp_theme_path: Path) -> None:
    manager = ThemeManager()
    manager.save_custom_theme("Brand", VALID_COLORS, apply_immediately=True)

    assert manager.delete_custom_theme("Brand") is True
    assert manager.get_current_theme_name() == "Light"
    assert "Brand" not in manager.custom_themes


def test_get_custom_theme_path_uses_app_config_location(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_settings: type[FakeSettings],
) -> None:
    class FakeQStandardPaths:
        class StandardLocation:
            AppConfigLocation = object()

        @staticmethod
        def writableLocation(location: object) -> str:
            assert location is FakeQStandardPaths.StandardLocation.AppConfigLocation
            return str(tmp_path / "config")

    monkeypatch.setattr(theme_module, "QStandardPaths", FakeQStandardPaths)

    path = ThemeManager()._get_custom_theme_path()

    assert path == tmp_path / "config" / "user_themes.json"
    assert path.parent.is_dir()
