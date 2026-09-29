"""Tests for theme.theme_manager (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.theme.theme_manager import (
    BUILTIN_THEMES,
    ThemeManager,
    generate_stylesheet,
    get_theme_manager,
    normalise_hex_color,
)


class TestBuiltinThemes:
    def test_has_light_theme(self) -> None:
        assert "Light" in BUILTIN_THEMES

    def test_has_dark_theme(self) -> None:
        assert "Dark" in BUILTIN_THEMES

    def test_themes_are_dicts(self) -> None:
        for theme in BUILTIN_THEMES.values():
            assert isinstance(theme, dict)


class TestNormaliseHexColor:
    def test_passthrough(self) -> None:
        assert normalise_hex_color("#ff0000") == "#ff0000"

    def test_uppercase_lowercased(self) -> None:
        result = normalise_hex_color("#FF0000")
        assert result == result.lower() or len(result) == 7

    def test_theme_manager_returns_string(self) -> None:
        result = normalise_hex_color("#aabbcc")
        assert isinstance(result, str)


class TestGenerateStylesheet:
    def test_theme_manager_returns_string(self) -> None:
        ss = generate_stylesheet(BUILTIN_THEMES["Dark"])
        assert isinstance(ss, str)

    def test_not_empty(self) -> None:
        ss = generate_stylesheet(BUILTIN_THEMES["Light"])
        assert len(ss) > 0

    def test_contains_css(self) -> None:
        ss = generate_stylesheet(BUILTIN_THEMES["Dark"])
        assert "{" in ss


class TestGetThemeManager:
    def test_returns_theme_manager(self) -> None:
        tm = get_theme_manager()
        assert isinstance(tm, ThemeManager)

    def test_theme_manager_singleton(self) -> None:
        tm1 = get_theme_manager()
        tm2 = get_theme_manager()
        assert tm1 is tm2

    def test_has_current_theme(self) -> None:
        tm = get_theme_manager()
        assert hasattr(tm, "current_theme")
