"""Tests for sidekick.ui.catppuccin_theme (Issues #1949, #1744)."""

from __future__ import annotations

from sidekick.ui.catppuccin_theme import (
    COLORS,
    get_stylesheet,
)


class TestCatppuccinColors:
    def test_colors_is_dict(self) -> None:
        assert isinstance(COLORS, dict)

    def test_colors_not_empty(self) -> None:
        assert len(COLORS) > 0

    def test_has_base_color(self) -> None:
        assert "base" in COLORS

    def test_has_text_color(self) -> None:
        assert "text" in COLORS


class TestGetStylesheet:
    def test_catppuccin_theme_returns_string(self) -> None:
        ss = get_stylesheet()
        assert isinstance(ss, str)

    def test_stylesheet_not_empty(self) -> None:
        ss = get_stylesheet()
        assert len(ss) > 0

    def test_stylesheet_contains_css(self) -> None:
        ss = get_stylesheet()
        assert "{" in ss and "}" in ss
