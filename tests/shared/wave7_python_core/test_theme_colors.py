"""Tests for src/shared/python/theme/colors.py — color utilities and themes."""

from __future__ import annotations

import pytest

from src.shared.python.theme import colors

# ── is_valid_hex_color ─────────────────────────────────────────────────────


class TestIsValidHexColor:
    @pytest.mark.parametrize(
        "value",
        [
            "#fff",
            "#FFFF",
            "#ffffff",
            "#ffffffff",
            "fff",
            "ffffff",
            "#0A84FF",
            "#FF000080",
            "  #ff0000  ",
        ],
    )
    def test_valid(self, value: str) -> None:
        assert colors.is_valid_hex_color(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "   ",
            "#",
            "#gg",
            "#ggg",
            "#12345",
            "#1234567",
            "not-a-color",
            "rgb(1,2,3)",
            "#fffffffff",
        ],
    )
    def test_invalid(self, value: str) -> None:
        assert colors.is_valid_hex_color(value) is False


# ── normalise_hex_color ────────────────────────────────────────────────────


class TestNormaliseHexColor:
    def test_three_digit_expansion(self) -> None:
        assert colors.normalise_hex_color("#f00") == "#ff0000"
        assert colors.normalise_hex_color("f00") == "#ff0000"

    def test_six_digit_lowercased(self) -> None:
        assert colors.normalise_hex_color("#FF0000") == "#ff0000"

    def test_strip_whitespace(self) -> None:
        assert colors.normalise_hex_color("  #ABCDEF  ") == "#abcdef"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid colour"):
            colors.normalise_hex_color("not-a-color")


# ── get_rgba ───────────────────────────────────────────────────────────────


class TestGetRgba:
    def test_six_digit(self) -> None:
        assert colors.get_rgba("#ff0000") == (1.0, 0.0, 0.0, 1.0)

    def test_six_digit_no_hash(self) -> None:
        assert colors.get_rgba("00ff00") == (0.0, 1.0, 0.0, 1.0)

    def test_alpha_parameter(self) -> None:
        r, g, b, a = colors.get_rgba("#ff0000", alpha=0.5)
        assert (r, g, b) == (1.0, 0.0, 0.0)
        assert a == 0.5

    def test_eight_digit_alpha_channel(self) -> None:
        # 80 hex == 128/255 ~ 0.502
        r, g, b, a = colors.get_rgba("#ff000080")
        assert (r, g, b) == (1.0, 0.0, 0.0)
        assert pytest.approx(a, rel=1e-3) == 128 / 255

    def test_eight_digit_alpha_multiplied(self) -> None:
        # alpha kwarg multiplies hex alpha
        _, _, _, a = colors.get_rgba("#ff000080", alpha=0.5)
        assert pytest.approx(a, rel=1e-3) == (128 / 255) * 0.5

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="hex_color must be provided"):
            colors.get_rgba(None)  # type: ignore[arg-type]

    def test_wrong_length_raises(self) -> None:
        # Bug fix: previously the duplicated dead None check let through wrong
        # lengths and produced a misleading ValueError from int(). Now we get
        # an explicit length-validation error.
        with pytest.raises(ValueError, match="6 or 8 hex digits"):
            colors.get_rgba("#fff")


# ── _is_dark_theme / is_dark_theme ─────────────────────────────────────────


class TestIsDarkTheme:
    def test_dark_theme_bg(self) -> None:
        assert colors._is_dark_theme({"bg": "#000000"}) is True

    def test_light_theme_bg(self) -> None:
        assert colors._is_dark_theme({"bg": "#ffffff"}) is False

    def test_missing_bg_defaults_light(self) -> None:
        assert colors._is_dark_theme({}) is False

    def test_short_bg_returns_false(self) -> None:
        # Less-than-6-hex bg falls into the "return False" branch.
        assert colors._is_dark_theme({"bg": "#fff"}) is False

    def test_is_dark_by_name(self) -> None:
        assert colors.is_dark_theme("Dark") is True
        assert colors.is_dark_theme("Light") is False

    def test_unknown_theme_name(self) -> None:
        assert colors.is_dark_theme("NoSuchTheme") is False


# ── get_matplotlib_colors ──────────────────────────────────────────────────


class TestGetMatplotlibColors:
    def test_dark_theme(self) -> None:
        theme = colors.BUILTIN_THEMES["Dark"]
        out = colors.get_matplotlib_colors(theme)
        assert out["figure.facecolor"] == theme["bg"]
        assert out["axes.facecolor"] == theme["group_bg"]
        # Dark themes use lower grid alpha
        assert out["grid.alpha"] == 0.3

    def test_light_theme(self) -> None:
        theme = colors.BUILTIN_THEMES["Light"]
        out = colors.get_matplotlib_colors(theme)
        assert out["grid.alpha"] == 0.5


# ── module constants ───────────────────────────────────────────────────────


class TestModuleConstants:
    def test_builtin_themes_loaded(self) -> None:
        assert "Light" in colors.BUILTIN_THEMES
        assert "Dark" in colors.BUILTIN_THEMES

    def test_required_keys_present(self) -> None:
        for name, theme in colors.BUILTIN_THEMES.items():
            for key in colors.THEME_COLOR_KEYS:
                assert key in theme, f"Theme {name!r} missing key {key!r}"

    def test_chart_colors_non_empty(self) -> None:
        assert len(colors.CHART_COLORS) >= 6
        for c in colors.CHART_COLORS:
            assert colors.is_valid_hex_color(c)
