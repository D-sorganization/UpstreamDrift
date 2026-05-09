"""Tests for plot_theme.integration module (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.plot_theme.integration import (
    PlotTheme,
    PlotThemeManager,
    apply_plot_theme,
    get_plot_theme_manager,
    get_theme,
    get_theme_colors,
)


class TestGetPlotThemeManager:
    def test_returns_manager(self) -> None:
        mgr = get_plot_theme_manager()
        assert isinstance(mgr, PlotThemeManager)

    def test_plot_theme_integration_singleton(self) -> None:
        mgr1 = get_plot_theme_manager()
        mgr2 = get_plot_theme_manager()
        assert mgr1 is mgr2


class TestGetTheme:
    def test_returns_plot_theme(self) -> None:
        theme = get_theme("nord")
        assert isinstance(theme, PlotTheme)

    def test_different_themes(self) -> None:
        t1 = get_theme("nord")
        t2 = get_theme("dracula")
        assert t1 is not t2

    def test_invalid_theme_raises(self) -> None:
        import pytest

        with pytest.raises(KeyError):
            get_theme("nonexistent_theme_xyz")


class TestGetThemeColors:
    def test_plot_theme_integration_returns_dict(self) -> None:
        colors = get_theme_colors()
        assert isinstance(colors, dict)

    def test_has_entries(self) -> None:
        colors = get_theme_colors()
        assert len(colors) > 0


class TestApplyPlotTheme:
    def test_callable(self) -> None:
        assert callable(apply_plot_theme)
