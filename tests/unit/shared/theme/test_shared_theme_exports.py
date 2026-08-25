"""Unit tests for src.shared.python.theme package exports (Issue #9037)."""

from __future__ import annotations

import pytest

import src.shared.python.theme as theme
from src.shared.python.theme import (
    CSS_FONT_DISPLAY,
    CSS_FONT_MONO,
    CSS_FONT_UI,
    DARK_THEME,
    FONT_STACK_DISPLAY,
    FONT_STACK_MONO,
    FONT_STACK_UI,
    Colors,
    FontSizes,
    FontWeights,
    Sizes,
    ThemePalette,
    Weights,
    get_current_colors,
    get_display_font,
    get_mono_font,
    get_qfont,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_theme_exports_available() -> None:
    """Verify all required symbols are exported from src.shared.python.theme."""
    assert hasattr(theme, "Colors")
    assert hasattr(theme, "ThemePalette")
    assert hasattr(theme, "DARK_THEME")
    assert hasattr(theme, "get_current_colors")
    assert hasattr(theme, "Sizes")
    assert hasattr(theme, "Weights")
    assert hasattr(theme, "FontSizes")
    assert hasattr(theme, "FontWeights")
    assert hasattr(theme, "get_qfont")
    assert hasattr(theme, "get_display_font")
    assert hasattr(theme, "get_mono_font")
    assert hasattr(theme, "CSS_FONT_DISPLAY")
    assert hasattr(theme, "CSS_FONT_UI")
    assert hasattr(theme, "CSS_FONT_MONO")
    assert hasattr(theme, "FONT_STACK_DISPLAY")
    assert hasattr(theme, "FONT_STACK_UI")
    assert hasattr(theme, "FONT_STACK_MONO")


def test_colors_export_is_dynamic() -> None:
    """Colors exported at top level dynamically resolves tokens."""
    assert Colors is theme.Colors
    assert isinstance(Colors.PRIMARY, str)
    assert isinstance(Colors.BG_BASE, str)
    assert Colors.PRIMARY.startswith("#")
    assert Colors.BG_BASE.startswith("#")


def test_font_sizes_and_weights_constants() -> None:
    """Sizes and Weights instances provide standard typography values."""
    assert Sizes.BASE == 10
    assert Sizes.SM == 9
    assert Sizes.LG == 13
    assert Sizes.XL == 16
    assert Sizes.XXL == 24
    assert Sizes.XXXL == 32

    assert Weights.NORMAL == 400
    assert Weights.BOLD == 700
    assert Weights.MEDIUM == 500
    assert Weights.SEMIBOLD == 600


def test_font_stacks_and_css_strings() -> None:
    """CSS font strings and font stacks are valid strings."""
    assert "Outfit" in FONT_STACK_UI
    assert "JetBrains Mono" in FONT_STACK_MONO
    assert "Outfit" in FONT_STACK_DISPLAY

    assert CSS_FONT_UI.startswith("font-family:")
    assert CSS_FONT_MONO.startswith("font-family:")
    assert CSS_FONT_DISPLAY.startswith("font-family:")


def test_palette_helpers() -> None:
    """ThemePalette and get_current_colors return correct types."""
    palette = get_current_colors()
    assert isinstance(palette, ThemePalette)
    assert isinstance(DARK_THEME, ThemePalette)
    assert hasattr(palette, "bg")
