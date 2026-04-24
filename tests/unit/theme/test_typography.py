"""Tests for src.shared.python.theme.typography (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.theme.typography import (
    CSS_FONT_DISPLAY,
    CSS_FONT_MONO,
    CSS_FONT_UI,
    FONT_STACK_DISPLAY,
    FONT_STACK_MONO,
    FONT_STACK_UI,
    FontSizes,
    FontWeights,
    Sizes,
    Weights,
)


class TestFontSizes:
    def test_xs_is_smallest(self) -> None:
        assert FontSizes.XS < FontSizes.SM

    def test_ordered_hierarchy(self) -> None:
        sizes = [
            FontSizes.XS,
            FontSizes.SM,
            FontSizes.BASE,
            FontSizes.MD,
            FontSizes.LG,
            FontSizes.XL,
            FontSizes.XXL,
            FontSizes.XXXL,
        ]
        assert sizes == sorted(sizes)

    def test_sizes_instance_accessible(self) -> None:
        # Sizes is a module-level instance; class vars should still be accessible
        assert FontSizes.BASE == 10

    def test_xxxl_is_largest(self) -> None:
        assert FontSizes.XXXL >= FontSizes.XXL


class TestFontWeights:
    def test_normal_weight(self) -> None:
        assert FontWeights.NORMAL == 400

    def test_bold_weight(self) -> None:
        assert FontWeights.BOLD == 700

    def test_thin_is_lightest(self) -> None:
        assert FontWeights.THIN < FontWeights.LIGHT

    def test_ordered_hierarchy(self) -> None:
        weights = [
            FontWeights.THIN,
            FontWeights.LIGHT,
            FontWeights.NORMAL,
            FontWeights.MEDIUM,
            FontWeights.SEMIBOLD,
            FontWeights.BOLD,
            FontWeights.EXTRABOLD,
        ]
        assert weights == sorted(weights)


class TestFontStacks:
    def test_ui_stack_is_string(self) -> None:
        assert isinstance(FONT_STACK_UI, str)

    def test_mono_stack_is_string(self) -> None:
        assert isinstance(FONT_STACK_MONO, str)

    def test_display_stack_is_string(self) -> None:
        assert isinstance(FONT_STACK_DISPLAY, str)

    def test_ui_stack_has_fallback(self) -> None:
        assert "sans-serif" in FONT_STACK_UI

    def test_mono_stack_has_monospace_fallback(self) -> None:
        assert "monospace" in FONT_STACK_MONO

    def test_display_stack_has_sans_fallback(self) -> None:
        assert "sans-serif" in FONT_STACK_DISPLAY


class TestCSSStrings:
    def test_css_font_ui_is_string(self) -> None:
        assert isinstance(CSS_FONT_UI, str)

    def test_css_font_ui_has_font_family(self) -> None:
        assert "font-family" in CSS_FONT_UI

    def test_css_font_mono_has_font_family(self) -> None:
        assert "font-family" in CSS_FONT_MONO

    def test_css_font_display_has_font_family(self) -> None:
        assert "font-family" in CSS_FONT_DISPLAY


class TestModuleLevelInstances:
    def test_sizes_is_font_sizes(self) -> None:
        assert isinstance(Sizes, FontSizes)

    def test_weights_is_font_weights(self) -> None:
        assert isinstance(Weights, FontWeights)
