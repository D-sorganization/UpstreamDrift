"""Unit tests for src.shared.python.theme.palette (Issues #8972, #9037)."""

from __future__ import annotations

import pytest

from src.shared.python.theme.colors import BUILTIN_THEMES
from src.shared.python.theme.palette import (
    DARK_THEME,
    SEMANTIC_ALIASES,
    Colors,
    ThemePalette,
    get_current_colors,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class TestThemePalette:
    def test_dict_access(self) -> None:
        palette = ThemePalette({"bg": "#111111", "accent": "#222222"})
        assert palette["bg"] == "#111111"
        assert palette["accent"] == "#222222"

    def test_canonical_key_attribute_access(self) -> None:
        palette = ThemePalette({"bg": "#111111", "accent": "#222222"})
        assert palette.bg == "#111111"
        assert palette.accent == "#222222"

    def test_case_insensitive_attribute_access(self) -> None:
        palette = ThemePalette({"bg": "#111111", "accent": "#222222"})
        assert palette.BG == "#111111"
        assert palette.ACCENT == "#222222"

    def test_semantic_alias_attribute_access(self) -> None:
        palette = ThemePalette(
            {
                "bg": "#111111",
                "group_bg": "#222222",
                "border": "#333333",
                "accent": "#444444",
                "text": "#555555",
                "button_hover": "#666666",
            }
        )
        assert palette.bg_base == "#111111"
        assert palette.BG_BASE == "#111111"
        assert palette.primary == "#444444"
        assert palette.PRIMARY == "#444444"
        assert palette.text_primary == "#555555"
        assert palette.TEXT_PRIMARY == "#555555"
        assert palette.bg_elevated == "#222222"
        assert palette.BG_ELEVATED == "#222222"
        assert palette.border_default == "#333333"
        assert palette.BORDER_DEFAULT == "#333333"
        assert palette.primary_hover == "#666666"
        assert palette.PRIMARY_HOVER == "#666666"

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        palette = ThemePalette({"bg": "#111111"})
        with pytest.raises(AttributeError, match="has no color 'nonexistent_color'"):
            _ = palette.nonexistent_color

    def test_get_current_colors_classmethod(self) -> None:
        palette = ThemePalette.get_current_colors()
        assert isinstance(palette, ThemePalette)
        assert hasattr(palette, "bg")


class TestColorsDynamicResolution:
    def test_default_tokens_resolve(self) -> None:
        assert hasattr(Colors, "PRIMARY")
        assert hasattr(Colors, "BG_BASE")
        assert hasattr(Colors, "TEXT_PRIMARY")
        assert hasattr(Colors, "ACCENT")
        assert hasattr(Colors, "BORDER_DEFAULT")
        assert hasattr(Colors, "SUCCESS")
        assert hasattr(Colors, "WARNING")
        assert hasattr(Colors, "ERROR")
        assert hasattr(Colors, "INFO")
        assert not hasattr(Colors, "COMPLETELY_INVALID_KEY")

    def test_dynamic_theme_switching_updates_colors_tokens(self) -> None:
        try:
            from src.shared.python.theme.theme_manager import ThemeManager
        except ImportError:
            pytest.skip("PyQt6 ThemeManager unavailable in this environment")

        manager = ThemeManager.instance()
        original_theme = manager.get_current_theme_name()

        try:
            # Switch to Dark theme
            manager.change_theme("Dark")
            dark_primary = Colors.PRIMARY
            dark_bg = Colors.BG_BASE
            dark_text = Colors.TEXT_PRIMARY
            assert dark_primary == BUILTIN_THEMES["Dark"]["accent"]
            assert dark_bg == BUILTIN_THEMES["Dark"]["bg"]
            assert dark_text == BUILTIN_THEMES["Dark"]["text"]

            # Switch to Light theme
            manager.change_theme("Light")
            light_primary = Colors.PRIMARY
            light_bg = Colors.BG_BASE
            light_text = Colors.TEXT_PRIMARY
            assert light_primary == BUILTIN_THEMES["Light"]["accent"]
            assert light_bg == BUILTIN_THEMES["Light"]["bg"]
            assert light_text == BUILTIN_THEMES["Light"]["text"]

            # Tokens must differ between Dark and Light
            assert dark_primary != light_primary
            assert dark_bg != light_bg
            assert dark_text != light_text

            # Switch to Ocean Blue
            if "Ocean Blue" in BUILTIN_THEMES:
                ocean_blue = BUILTIN_THEMES["Ocean Blue"]
                manager.change_theme("Ocean Blue")
                assert ocean_blue["accent"] == Colors.PRIMARY
                assert ocean_blue["bg"] == Colors.BG_BASE

            # Switch to Forest Green
            if "Forest Green" in BUILTIN_THEMES:
                forest_green = BUILTIN_THEMES["Forest Green"]
                manager.change_theme("Forest Green")
                assert forest_green["accent"] == Colors.PRIMARY
                assert forest_green["bg"] == Colors.BG_BASE
        finally:
            manager.change_theme(original_theme)

    def test_colors_instance_access_matches_class_access(self) -> None:
        instance = Colors()
        assert instance.PRIMARY == Colors.PRIMARY
        assert instance.BG_BASE == Colors.BG_BASE
        assert instance.TEXT_PRIMARY == Colors.TEXT_PRIMARY

    def test_colors_get_current_colors(self) -> None:
        palette = Colors.get_current_colors()
        assert isinstance(palette, ThemePalette)
        assert palette.PRIMARY == Colors.PRIMARY


def test_dark_theme_fallback_constants() -> None:
    assert isinstance(DARK_THEME, ThemePalette)
    assert DARK_THEME.bg == BUILTIN_THEMES["Dark"]["bg"]
    assert DARK_THEME.accent == BUILTIN_THEMES["Dark"]["accent"]
    assert "bg_base" in SEMANTIC_ALIASES
    assert "primary" in SEMANTIC_ALIASES
