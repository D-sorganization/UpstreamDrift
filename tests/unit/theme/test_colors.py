"""Tests for src.shared.python.theme.colors (Issues #1949, #1744)."""

from __future__ import annotations

import pytest

from src.shared.python.theme.colors import (
    BUILTIN_THEMES,
    SEMANTIC_COLOR_KEYS,
    THEME_COLOR_KEYS,
    Colors,
    _is_dark_theme,
    get_matplotlib_colors,
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

    @pytest.mark.parametrize(
        "value",
        [
            "abc",
            "#abcd",
            "11223344",
            "  #ABCDEF  ",
        ],
    )
    def test_valid_edge_formats(self, value: str) -> None:
        assert is_valid_hex_color(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "   ",
            "#12",
            "#12345",
            "#1234567",
            "#123456789",
            "not-a-color",
        ],
    )
    def test_invalid_edge_formats(self, value: str) -> None:
        assert is_valid_hex_color(value) is False


class TestNormaliseHexColor:
    def test_three_digit_expands_to_seven_chars(self) -> None:
        result = normalise_hex_color("#ABC")
        assert len(result) == 7  # #AABBCC
        assert result.startswith("#")

    def test_six_digit_preserved(self) -> None:
        result = normalise_hex_color("#aabbcc")
        assert len(result) == 7
        assert result.startswith("#")

    def test_strips_whitespace_and_lowercases(self) -> None:
        assert normalise_hex_color("  AABBCC  ") == "#aabbcc"

    def test_four_digit_alpha_format_is_preserved(self) -> None:
        assert normalise_hex_color("#AbCd") == "#abcd"

    def test_eight_digit_alpha_format_is_preserved(self) -> None:
        assert normalise_hex_color("AABBCCDD") == "#aabbccdd"

    @pytest.mark.parametrize("value", ["", "#12", "#xyzxyz", "#12345"])
    def test_invalid_values_raise(self, value: str) -> None:
        with pytest.raises(ValueError, match="Invalid colour value"):
            normalise_hex_color(value)


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

    def test_eight_digit_hex_uses_embedded_alpha(self) -> None:
        assert get_rgba("#ff000080") == (1.0, 0.0, 0.0, 128 / 255)

    def test_explicit_alpha_multiplies_embedded_alpha(self) -> None:
        _, _, _, a = get_rgba("0000ff80", alpha=0.5)
        assert a == pytest.approx((128 / 255) * 0.5)

    @pytest.mark.parametrize("value", [None, "#fff", "#12345", "#123456789"])
    def test_invalid_lengths_raise_value_error(self, value: str | None) -> None:
        with pytest.raises(ValueError):
            get_rgba(value)  # type: ignore[arg-type]

    def test_invalid_hex_digits_raise_value_error(self) -> None:
        with pytest.raises(ValueError):
            get_rgba("#zzzzzz")


class TestThemeColorKeys:
    def test_is_tuple(self) -> None:
        assert isinstance(THEME_COLOR_KEYS, tuple)

    def test_contains_bg(self) -> None:
        assert "bg" in THEME_COLOR_KEYS

    def test_contains_text(self) -> None:
        assert "text" in THEME_COLOR_KEYS

    def test_colors_nonempty(self) -> None:
        assert len(THEME_COLOR_KEYS) > 0


class TestSemanticColorKeys:
    def test_semantic_keys_include_expected_roles(self) -> None:
        assert set(SEMANTIC_COLOR_KEYS) == {
            "success",
            "warning",
            "error",
            "info",
            "link",
            "link_hover",
            "selection_bg",
            "selection_text",
        }


class TestIsDarkTheme:
    def test_colors_returns_bool(self) -> None:
        result = is_dark_theme("dark")
        assert isinstance(result, bool)

    def test_builtin_dark_theme_detected(self) -> None:
        assert is_dark_theme("Dark") is True

    def test_builtin_light_theme_not_dark(self) -> None:
        assert is_dark_theme("Light") is False

    def test_unknown_theme_not_dark(self) -> None:
        assert is_dark_theme("missing") is False

    @pytest.mark.parametrize(
        ("theme", "expected"),
        [
            ({"bg": "#000000"}, True),
            ({"bg": "#ffffff"}, False),
            ({"bg": "#7f7f7f"}, True),
            ({"bg": "bad"}, False),
            ({}, False),
        ],
    )
    def test_private_dark_theme_edges(
        self, theme: dict[str, str], expected: bool
    ) -> None:
        assert _is_dark_theme(theme) is expected


class TestGetMatplotlibColors:
    def test_dark_theme_uses_lower_grid_alpha(self) -> None:
        colors = get_matplotlib_colors(BUILTIN_THEMES["Dark"])
        assert colors["figure.facecolor"] == BUILTIN_THEMES["Dark"]["bg"]
        assert colors["grid.alpha"] == 0.3

    def test_light_theme_uses_higher_grid_alpha(self) -> None:
        colors = get_matplotlib_colors(BUILTIN_THEMES["Light"])
        assert colors["axes.facecolor"] == BUILTIN_THEMES["Light"]["group_bg"]
        assert colors["grid.alpha"] == 0.5


class TestColorsClass:
    def test_colors_attributes_exist(self) -> None:
        assert hasattr(Colors, "BG_BASE")
        assert Colors.BG_BASE == "#1a1d23"
        assert hasattr(Colors, "PRIMARY")
        assert Colors.PRIMARY == "#4a7ba7"
        assert hasattr(Colors, "PRIMARY_HOVER")
        assert Colors.PRIMARY_HOVER == "#5a8fc4"
        assert hasattr(Colors, "SUCCESS")
        assert Colors.SUCCESS == "#30d158"
