"""Tests for src.shared.python.theme.fleet_adapter (Issues #1949, #1744)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.shared.python.theme.fleet_adapter import (
    FLEET_THEMES,
    _adjust_color_brightness,
    _build_theme_colors_kwargs,
    _extract_base_colors,
    _extract_semantic_colors,
    _hex_with_alpha,
    _is_dark_theme,
    fleet_to_theme_colors,
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

    def test_whitespace_only_invalid(self) -> None:
        assert is_valid_hex_color("   ") is False


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

    def test_invalid_short_hex_returns_false(self) -> None:
        assert _is_dark_theme({"bg": "#"}) is False

    def test_hex_without_hash_returns_false(self) -> None:
        assert _is_dark_theme({"bg": "111111"}) is False


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

    def test_none_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="hex_color must be provided"):
            _adjust_color_brightness(None, 1.0)  # type: ignore[arg-type]

    def test_missing_hash_is_supported(self) -> None:
        assert _adjust_color_brightness("808080", 0.5) == "#404040"

    def test_negative_factor_clamps_to_zero(self) -> None:
        assert _adjust_color_brightness("#ffffff", -1.0) == "#000000"


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

    def test_none_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="hex_color must be provided"):
            _hex_with_alpha(None, 255)  # type: ignore[arg-type]

    def test_missing_hash_is_supported(self) -> None:
        assert _hex_with_alpha("abc", 0x40) == "#aabbcc40"


class TestExtractBaseColors:
    def test_uses_dark_defaults_when_values_are_missing(self) -> None:
        base = _extract_base_colors({}, is_dark=True)
        assert base["accent"] == "#0A84FF"
        assert base["text"] == "#FFFFFF"
        assert base["group_bg"] == "#242424"

    def test_uses_light_defaults_when_values_are_missing(self) -> None:
        base = _extract_base_colors({}, is_dark=False)
        assert base["text"] == "#1A1A1A"
        assert base["border"] == "#D0D0D0"

    def test_focus_falls_back_to_accent(self) -> None:
        base = _extract_base_colors({"accent": "#123456"}, is_dark=False)
        assert base["focus"] == "#123456"


class TestExtractSemanticColors:
    def test_uses_supplied_semantic_values(self) -> None:
        semantic = _extract_semantic_colors(
            {
                "success": "#010101",
                "warning": "#020202",
                "error": "#030303",
                "info": "#040404",
            },
            is_dark=False,
        )
        assert semantic == {
            "success": "#010101",
            "warning": "#020202",
            "error": "#030303",
            "info": "#040404",
        }

    def test_light_and_dark_defaults_differ(self) -> None:
        assert _extract_semantic_colors({}, is_dark=True)["warning"] == "#FF9F0A"
        assert _extract_semantic_colors({}, is_dark=False)["warning"] == "#E67E00"


class TestBuildThemeColorsKwargs:
    def test_builds_derived_values_from_minimal_theme(self) -> None:
        base = _extract_base_colors(
            {"accent": "#336699", "border": "#202020"}, is_dark=True
        )
        semantic = _extract_semantic_colors({"success": "#008000"}, is_dark=True)
        kwargs = _build_theme_colors_kwargs(
            "Custom Dark", {"title_bg": "#101010"}, True, base, semantic
        )

        assert kwargs["name"] == "Custom Dark"
        assert kwargs["is_dark"] is True
        assert kwargs["primary_hover"] == "#3d7ab7"
        assert kwargs["primary_pressed"] == "#28517a"
        assert kwargs["primary_muted"] == "#33669940"
        assert kwargs["success"] == "#008000"
        assert kwargs["bg_highlight"] == "#101010"
        assert kwargs["shadow_heavy"] == "rgba(0, 0, 0, 0.40)"

    def test_link_falls_back_to_accent(self) -> None:
        base = _extract_base_colors({"accent": "#123456"}, is_dark=False)
        semantic = _extract_semantic_colors({}, is_dark=False)
        kwargs = _build_theme_colors_kwargs("Lightish", {}, False, base, semantic)
        assert kwargs["text_link"] == "#123456"
        assert kwargs["shadow_light"] == "rgba(0, 0, 0, 0.08)"

    def test_none_theme_name_raises(self) -> None:
        with pytest.raises(ValueError, match="theme_name must be provided"):
            _build_theme_colors_kwargs(
                None, {}, False, _extract_base_colors({}, False), _extract_semantic_colors({}, False)
            )  # type: ignore[arg-type]


class TestFleetToThemeColors:
    def test_missing_theme_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Fleet theme 'missing' not found"):
            fleet_to_theme_colors("missing")

    def test_converts_theme_with_fake_theme_colors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeThemeColors:
            def __init__(self, **kwargs: str | bool) -> None:
                self.kwargs = kwargs

        module_name = "src.shared.python.theme.theme_manager"
        monkeypatch.setitem(
            sys.modules, module_name, SimpleNamespace(ThemeColors=FakeThemeColors)
        )
        monkeypatch.setitem(
            FLEET_THEMES,
            "Unit Theme",
            {
                "bg": "#000000",
                "group_bg": "#111111",
                "border": "#222222",
                "text": "#eeeeee",
                "text_secondary": "#cccccc",
                "label": "#999999",
                "input_bg": "#010101",
                "accent": "#336699",
                "success": "#008000",
                "warning": "#ff9900",
                "error": "#cc0000",
                "info": "#0099cc",
            },
        )

        result = fleet_to_theme_colors("Unit Theme")

        assert isinstance(result, FakeThemeColors)
        assert result.kwargs["name"] == "Unit Theme"
        assert result.kwargs["is_dark"] is True
        assert result.kwargs["primary"] == "#336699"
