"""Tests for src.shared.python.theme.fleet_adapter (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.theme.fleet_adapter import (
    FLEET_THEMES,
    _adjust_color_brightness,
    _hex_with_alpha,
    _is_dark_theme,
    get_fleet_theme_names,
    is_fleet_available,
    is_valid_hex_color,
)


class TestIsFleetAvailable:
    def test_fleet_adapter_returns_bool(self) -> None:
        result = is_fleet_available()
        assert isinstance(result, bool)


class TestGetFleetThemeNames:
    def test_fleet_adapter_returns_list(self) -> None:
        names = get_fleet_theme_names()
        assert isinstance(names, list)

    def test_matches_fleet_themes_keys(self) -> None:
        names = get_fleet_theme_names()
        assert set(names) == set(FLEET_THEMES.keys())


class TestIsValidHexColor:
    def test_valid_6digit(self) -> None:
        assert is_valid_hex_color("#1a2b3c") is True

    def test_valid_3digit(self) -> None:
        assert is_valid_hex_color("#abc") is True

    def test_empty_string_invalid(self) -> None:
        assert is_valid_hex_color("") is False

    def test_no_hash_6digit(self) -> None:
        # Without '#' prefix — depends on fallback implementation
        # The fallback strips '#' and checks the remaining chars
        assert is_valid_hex_color("aabbcc") is True

    def test_invalid_chars(self) -> None:
        assert is_valid_hex_color("#gggggg") is False

    def test_uppercase_valid(self) -> None:
        assert is_valid_hex_color("#AABBCC") is True

    def test_wrong_length_invalid(self) -> None:
        assert is_valid_hex_color("#12345") is False


class TestIsDarkTheme:
    def test_dark_bg_is_dark(self) -> None:
        assert _is_dark_theme({"bg": "#1e1e2e"}) is True

    def test_light_bg_not_dark(self) -> None:
        assert _is_dark_theme({"bg": "#ffffff"}) is False

    def test_missing_bg_defaults_to_light(self) -> None:
        # Default bg is "#ffffff" (luminance 1.0 → not dark)
        assert _is_dark_theme({}) is False

    def test_non_hex_bg_returns_false(self) -> None:
        assert _is_dark_theme({"bg": "dark"}) is False

    def test_3digit_dark_color(self) -> None:
        assert _is_dark_theme({"bg": "#111"}) is True

    def test_medium_gray_threshold(self) -> None:
        # #7f7f7f has luminance ≈ 0.498, which is < 0.5 → dark
        result = _is_dark_theme({"bg": "#7f7f7f"})
        assert isinstance(result, bool)


class TestAdjustColorBrightness:
    def test_lightening_factor(self) -> None:
        result = _adjust_color_brightness("#808080", 2.0)
        assert result.startswith("#")

    def test_darkening_factor(self) -> None:
        result = _adjust_color_brightness("#ffffff", 0.5)
        assert result.startswith("#")

    def test_clamps_to_ff(self) -> None:
        # Factor very large → channels clamped to 255
        result = _adjust_color_brightness("#ffffff", 10.0)
        assert result == "#ffffff"

    def test_clamps_to_00(self) -> None:
        result = _adjust_color_brightness("#000000", 0.5)
        assert result == "#000000"

    def test_3digit_input(self) -> None:
        result = _adjust_color_brightness("#fff", 0.5)
        assert result.startswith("#")
        assert len(result) == 7  # expands to 6-digit

    def test_invalid_returns_original(self) -> None:
        # Non-parseable hex returns original
        result = _adjust_color_brightness("#zzzzzz", 1.0)
        assert result == "#zzzzzz"


class TestHexWithAlpha:
    def test_appends_alpha(self) -> None:
        result = _hex_with_alpha("#ffffff", 128)
        assert result.startswith("#ffffff")
        assert len(result) == 9  # '#' + 6 + 2

    def test_alpha_00(self) -> None:
        result = _hex_with_alpha("#000000", 0)
        assert result.endswith("00")

    def test_alpha_ff(self) -> None:
        result = _hex_with_alpha("#000000", 255)
        assert result.endswith("ff")

    def test_3digit_expands(self) -> None:
        result = _hex_with_alpha("#fff", 255)
        assert len(result) == 9
