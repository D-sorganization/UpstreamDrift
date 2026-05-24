"""Tests for runtime-checkable theme protocols."""

from __future__ import annotations

from src.shared.python.theme.protocols import (
    StylesheetGenerator,
    ThemeProvider,
    ThemeSwitcher,
)


class CompleteThemeProvider:
    current_theme_name = "dark"

    def get_available_themes(self) -> list[str]:
        return ["dark", "light"]

    def get_current_colors(self) -> dict[str, str]:
        return {"background": "#000000", "foreground": "#ffffff"}


class MissingCurrentThemeName:
    def get_available_themes(self) -> list[str]:
        return ["dark"]

    def get_current_colors(self) -> dict[str, str]:
        return {"background": "#000000"}


class CompleteThemeSwitcher:
    def __init__(self) -> None:
        self.selected: str | None = None

    def get_available_themes(self) -> list[str]:
        return ["dark", "light"]

    def change_theme(self, name: str) -> None:
        self.selected = name


class MissingChangeTheme:
    def get_available_themes(self) -> list[str]:
        return ["dark"]


class CompleteStylesheetGenerator:
    def generate(self, colors: dict[str, str]) -> str:
        return f"QWidget {{ color: {colors['foreground']}; }}"


class MissingGenerate:
    def render(self, colors: dict[str, str]) -> str:
        return f"unused {colors}"


def test_theme_provider_runtime_conformance() -> None:
    provider = CompleteThemeProvider()

    assert isinstance(provider, ThemeProvider)
    assert provider.get_available_themes() == ["dark", "light"]
    assert provider.get_current_colors()["foreground"] == "#ffffff"
    assert provider.current_theme_name == "dark"


def test_theme_provider_runtime_non_conformance() -> None:
    assert not isinstance(MissingCurrentThemeName(), ThemeProvider)


def test_theme_switcher_runtime_conformance() -> None:
    switcher = CompleteThemeSwitcher()

    assert isinstance(switcher, ThemeSwitcher)
    switcher.change_theme("light")
    assert switcher.selected == "light"
    assert switcher.get_available_themes() == ["dark", "light"]


def test_theme_switcher_runtime_non_conformance() -> None:
    assert not isinstance(MissingChangeTheme(), ThemeSwitcher)


def test_stylesheet_generator_runtime_conformance() -> None:
    generator = CompleteStylesheetGenerator()

    assert isinstance(generator, StylesheetGenerator)
    assert generator.generate({"foreground": "#123456"}) == "QWidget { color: #123456; }"


def test_stylesheet_generator_runtime_non_conformance() -> None:
    assert not isinstance(MissingGenerate(), StylesheetGenerator)
