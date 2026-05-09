"""Tests for src.shared.python.theme.colors (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.theme.colors import (
    THEME_COLOR_KEYS,
    get_rgba,
    is_dark_theme,
    is_valid_hex_color,
    normalise_hex_color,
)


class TestIsValidHexColor:
    def test_valid_six_digit_hex(self) -> None:
        assert is_valid_hex_color("#FF0000") is True

    def test_valid_three_digit_hex(self) -> None:
        assert is_valid_hex_color("#F00") is True

    def test_invalid_non_hex_chars(self) -> None:
        assert is_valid_hex_color("#ZZZZZZ") is False

    def test_valid_lowercase(self) -> None:
        assert is_valid_hex_color("#aabbcc") is True

    def test_colors_returns_bool(self) -> None:
        assert isinstance(is_valid_hex_color("#FF0000"), bool)


class TestNormaliseHexColor:
    def test_three_digit_expands_to_seven_chars(self) -> None:
        result = normalise_hex_color("#ABC")
        assert len(result) == 7  # #AABBCC
        assert result.startswith("#")

    def test_six_digit_preserved(self) -> None:
        result = normalise_hex_color("#aabbcc")
        assert len(result) == 7
        assert result.startswith("#")


class TestGetRgba:
    def test_black_returns_zeros(self) -> None:
        r, g, b, a = get_rgba("#000000")
        assert r == 0.0
        assert g == 0.0
        assert b == 0.0

    def test_white_returns_ones(self) -> None:
        r, g, b, a = get_rgba("#FFFFFF")
        assert abs(r - 1.0) < 1e-6
        assert abs(g - 1.0) < 1e-6
        assert abs(b - 1.0) < 1e-6

    def test_alpha_default_is_one(self) -> None:
        _, _, _, a = get_rgba("#FF0000")
        assert a == 1.0

    def test_custom_alpha(self) -> None:
        _, _, _, a = get_rgba("#FF0000", alpha=0.5)
        assert a == 0.5

    def test_returns_four_tuple(self) -> None:
        result = get_rgba("#123456")
        assert len(result) == 4


class TestThemeColorKeys:
    def test_is_tuple(self) -> None:
        assert isinstance(THEME_COLOR_KEYS, tuple)

    def test_contains_bg(self) -> None:
        assert "bg" in THEME_COLOR_KEYS

    def test_contains_text(self) -> None:
        assert "text" in THEME_COLOR_KEYS

    def test_colors_nonempty(self) -> None:
        assert len(THEME_COLOR_KEYS) > 0


class TestIsDarkTheme:
    def test_colors_returns_bool(self) -> None:
        result = is_dark_theme("dark")
        assert isinstance(result, bool)
