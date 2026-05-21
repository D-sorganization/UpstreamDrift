"""Tests for src.shared.python.theme.typography (Issues #1949, #1744)."""

from __future__ import annotations

import sys
import types

import pytest

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
    get_display_font,
    get_mono_font,
    get_qfont,
)


class FakeQFont:
    class Weight(int):
        pass

    def __init__(self) -> None:
        self.family: str | None = None
        self.families: list[str] = []
        self.point_size: int | None = None
        self.weight: FakeQFont.Weight | None = None
        self.italic = False

    def setFamily(self, family: str) -> None:
        self.family = family

    def setFamilies(self, families: list[str]) -> None:
        self.families = families

    def setPointSize(self, size: int) -> None:
        self.point_size = size

    def setWeight(self, weight: FakeQFont.Weight) -> None:
        self.weight = weight

    def setItalic(self, italic: bool) -> None:
        self.italic = italic


@pytest.fixture
def fake_qfont(monkeypatch: pytest.MonkeyPatch) -> type[FakeQFont]:
    pyqt_module = types.ModuleType("PyQt6")
    qtgui_module = types.ModuleType("PyQt6.QtGui")
    qtgui_module.QFont = FakeQFont
    pyqt_module.QtGui = qtgui_module
    monkeypatch.setitem(sys.modules, "PyQt6", pyqt_module)
    monkeypatch.setitem(sys.modules, "PyQt6.QtGui", qtgui_module)
    return FakeQFont


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

    def test_ui_stack_prioritizes_outfit(self) -> None:
        assert FONT_STACK_UI.split(",")[0].strip() == '"Outfit"'

    def test_mono_stack_is_string(self) -> None:
        assert isinstance(FONT_STACK_MONO, str)

    def test_display_stack_is_string(self) -> None:
        assert isinstance(FONT_STACK_DISPLAY, str)

    def test_display_stack_prioritizes_outfit(self) -> None:
        assert FONT_STACK_DISPLAY.split(",")[0].strip() == '"Outfit"'

    def test_ui_stack_has_fallback(self) -> None:
        assert "sans-serif" in FONT_STACK_UI

    def test_mono_stack_has_monospace_fallback(self) -> None:
        assert "monospace" in FONT_STACK_MONO

    def test_display_stack_has_sans_fallback(self) -> None:
        assert "sans-serif" in FONT_STACK_DISPLAY

    def test_ui_stack_parses_clean_family_names(self, fake_qfont: type[FakeQFont]) -> None:
        font = get_qfont()

        assert isinstance(font, fake_qfont)
        assert font.families[:3] == ["Outfit", "Inter", "SF Pro Display"]
        assert font.families[-1] == "sans-serif"

    def test_display_stack_parses_clean_family_names(
        self,
        fake_qfont: type[FakeQFont],
    ) -> None:
        font = get_display_font()

        assert isinstance(font, fake_qfont)
        assert font.families[:3] == ["Outfit", "SF Pro Display", "Inter"]
        assert font.families[-1] == "sans-serif"

    def test_mono_stack_parses_clean_family_names(
        self,
        fake_qfont: type[FakeQFont],
    ) -> None:
        font = get_mono_font()

        assert isinstance(font, fake_qfont)
        assert font.families[:3] == ["JetBrains Mono", "SF Mono", "Cascadia Code"]
        assert font.families[-1] == "monospace"


class TestCSSStrings:
    def test_css_font_ui_is_string(self) -> None:
        assert isinstance(CSS_FONT_UI, str)

    def test_css_font_ui_mentions_outfit(self) -> None:
        assert "Outfit" in CSS_FONT_UI

    def test_css_font_ui_has_font_family(self) -> None:
        assert "font-family" in CSS_FONT_UI

    def test_css_font_mono_has_font_family(self) -> None:
        assert "font-family" in CSS_FONT_MONO

    def test_css_font_display_has_font_family(self) -> None:
        assert "font-family" in CSS_FONT_DISPLAY

    def test_css_font_ui_wraps_exact_stack(self) -> None:
        assert f"font-family: {FONT_STACK_UI};" == CSS_FONT_UI

    def test_css_font_mono_wraps_exact_stack(self) -> None:
        assert f"font-family: {FONT_STACK_MONO};" == CSS_FONT_MONO

    def test_css_font_display_wraps_exact_stack(self) -> None:
        assert f"font-family: {FONT_STACK_DISPLAY};" == CSS_FONT_DISPLAY


class TestQFontHelpers:
    @pytest.mark.parametrize("size", [None, 0, -1, 10.5, "10"])
    def test_get_qfont_rejects_invalid_sizes(self, size: object) -> None:
        with pytest.raises(ValueError, match="size"):
            get_qfont(size=size)  # type: ignore[arg-type]

    def test_get_qfont_uses_explicit_family_and_style(
        self,
        fake_qfont: type[FakeQFont],
    ) -> None:
        font = get_qfont(size=13, weight=FontWeights.SEMIBOLD, family="Aptos", italic=True)

        assert isinstance(font, fake_qfont)
        assert font.family == "Aptos"
        assert font.families == []
        assert font.point_size == 13
        assert font.weight == fake_qfont.Weight(FontWeights.SEMIBOLD)
        assert font.italic is True

    def test_get_qfont_uses_ui_stack_when_family_is_omitted(
        self,
        fake_qfont: type[FakeQFont],
    ) -> None:
        font = get_qfont(size=FontSizes.MD, weight=FontWeights.MEDIUM)

        assert isinstance(font, fake_qfont)
        assert font.family is None
        assert font.families[0] == "Outfit"
        assert font.point_size == FontSizes.MD
        assert font.weight == fake_qfont.Weight(FontWeights.MEDIUM)
        assert font.italic is False

    @pytest.mark.parametrize("size", [None, 0, -4, 12.5, "12"])
    def test_get_display_font_rejects_invalid_sizes(self, size: object) -> None:
        with pytest.raises(ValueError, match="size"):
            get_display_font(size=size)  # type: ignore[arg-type]

    def test_get_display_font_sets_stack_size_and_weight(
        self,
        fake_qfont: type[FakeQFont],
    ) -> None:
        font = get_display_font(size=FontSizes.XXL, weight=FontWeights.EXTRABOLD)

        assert isinstance(font, fake_qfont)
        assert font.families[0] == "Outfit"
        assert font.point_size == FontSizes.XXL
        assert font.weight == fake_qfont.Weight(FontWeights.EXTRABOLD)

    @pytest.mark.parametrize("size", [None, 0, -2, 9.5, "9"])
    def test_get_mono_font_rejects_invalid_sizes(self, size: object) -> None:
        with pytest.raises(ValueError, match="size"):
            get_mono_font(size=size)  # type: ignore[arg-type]

    def test_get_mono_font_sets_stack_size_and_weight(
        self,
        fake_qfont: type[FakeQFont],
    ) -> None:
        font = get_mono_font(size=FontSizes.SM, weight=FontWeights.LIGHT)

        assert isinstance(font, fake_qfont)
        assert font.families[0] == "JetBrains Mono"
        assert font.point_size == FontSizes.SM
        assert font.weight == fake_qfont.Weight(FontWeights.LIGHT)


class TestModuleLevelInstances:
    def test_sizes_is_font_sizes(self) -> None:
        assert isinstance(Sizes, FontSizes)

    def test_weights_is_font_weights(self) -> None:
        assert isinstance(Weights, FontWeights)
