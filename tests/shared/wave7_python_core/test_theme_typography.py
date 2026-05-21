"""Tests for src/shared/python/theme/typography.py.

The Qt-bound helpers (``get_qfont``, ``get_display_font``, ``get_mono_font``)
need PyQt6 to actually create a font, so we only call them under a guard. The
pure-Python parts — font stacks, ``FontSizes``, ``FontWeights``, CSS strings —
have no Qt dependency.
"""

from __future__ import annotations

import pytest

from theme import typography as t


class TestFontStacks:
    def test_ui_stack_includes_fallback(self) -> None:
        assert "sans-serif" in t.FONT_STACK_UI
        assert "Outfit" in t.FONT_STACK_UI

    def test_mono_stack_includes_fallback(self) -> None:
        assert "monospace" in t.FONT_STACK_MONO

    def test_display_stack_includes_fallback(self) -> None:
        assert "sans-serif" in t.FONT_STACK_DISPLAY

    def test_css_strings(self) -> None:
        assert t.CSS_FONT_UI.startswith("font-family:")
        assert t.CSS_FONT_MONO.startswith("font-family:")
        assert t.CSS_FONT_DISPLAY.startswith("font-family:")


class TestFontSizes:
    def test_monotonic(self) -> None:
        sizes = [
            t.FontSizes.XS,
            t.FontSizes.SM,
            t.FontSizes.BASE,
            t.FontSizes.MD,
            t.FontSizes.LG,
            t.FontSizes.XL,
            t.FontSizes.XXL,
            t.FontSizes.XXXL,
        ]
        assert sizes == sorted(sizes)
        assert all(isinstance(s, int) and s > 0 for s in sizes)

    def test_convenience_singletons(self) -> None:
        assert t.Sizes.BASE == t.FontSizes.BASE


class TestFontWeights:
    def test_monotonic(self) -> None:
        weights = [
            t.FontWeights.THIN,
            t.FontWeights.LIGHT,
            t.FontWeights.NORMAL,
            t.FontWeights.MEDIUM,
            t.FontWeights.SEMIBOLD,
            t.FontWeights.BOLD,
            t.FontWeights.EXTRABOLD,
        ]
        assert weights == sorted(weights)

    def test_convenience_singletons(self) -> None:
        assert t.Weights.NORMAL == t.FontWeights.NORMAL


# Qt-dependent helpers — only run if PyQt6 is installed and a QApplication
# can be created.
PyQt6 = pytest.importorskip("PyQt6.QtWidgets")  # noqa: N816


@pytest.fixture(scope="module")
def qapp() -> object:
    """Create a single QApplication for the module (Qt requires exactly one)."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


class TestQFontHelpers:
    def test_get_qfont_defaults(self, qapp: object) -> None:
        font = t.get_qfont()
        assert font.pointSize() == t.FontSizes.BASE

    def test_get_qfont_custom(self, qapp: object) -> None:
        font = t.get_qfont(size=20, weight=t.FontWeights.BOLD, italic=True)
        assert font.pointSize() == 20
        assert font.italic() is True

    def test_get_qfont_named_family(self, qapp: object) -> None:
        font = t.get_qfont(family="Arial")
        assert font.family() == "Arial"

    def test_get_qfont_rejects_none_size(self, qapp: object) -> None:
        with pytest.raises(ValueError, match="size must be provided"):
            t.get_qfont(size=None)  # type: ignore[arg-type]

    def test_get_qfont_rejects_nonpositive_size(self, qapp: object) -> None:
        # Bug-fix coverage: previously the duplicated dead `not (size is not
        # None)` check accepted zero/negative sizes silently.
        with pytest.raises(ValueError, match="positive int"):
            t.get_qfont(size=0)

    def test_get_display_font(self, qapp: object) -> None:
        font = t.get_display_font(size=24)
        assert font.pointSize() == 24

    def test_get_display_font_rejects_zero(self, qapp: object) -> None:
        with pytest.raises(ValueError, match="positive int"):
            t.get_display_font(size=0)

    def test_get_mono_font(self, qapp: object) -> None:
        font = t.get_mono_font(size=12)
        assert font.pointSize() == 12

    def test_get_mono_font_rejects_zero(self, qapp: object) -> None:
        with pytest.raises(ValueError, match="positive int"):
            t.get_mono_font(size=0)
