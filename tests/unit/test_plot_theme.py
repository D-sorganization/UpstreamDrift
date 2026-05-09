"""Tests for plot_theme.manager, plot_theme.themes, and plot_theme.integration (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.plot_theme.manager import (
    PlotThemeManager,
    get_plot_theme_manager,
)
from src.shared.python.plot_theme.themes import (
    PlotTheme,
    get_theme,
    get_theme_names,
    register_theme,
)


class TestPlotTheme:
    def test_get_theme_names_returns_list(self) -> None:
        names = get_theme_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_theme_returns_plot_theme(self) -> None:
        names = get_theme_names()
        theme = get_theme(names[0])
        assert isinstance(theme, PlotTheme)

    def test_theme_has_figure_facecolor(self) -> None:
        theme = get_theme("catppuccin_mocha")
        assert hasattr(theme, "figure_facecolor")

    def test_theme_has_axes_facecolor(self) -> None:
        theme = get_theme("catppuccin_mocha")
        assert hasattr(theme, "axes_facecolor")

    def test_register_custom_theme(self) -> None:
        theme = get_theme(get_theme_names()[0])
        register_theme("test_theme_xyz", theme)
        assert "test_theme_xyz" in get_theme_names()

    def test_get_unknown_theme_raises(self) -> None:
        with pytest.raises((KeyError, ValueError)):
            get_theme("completely_unknown_theme_9999")


class TestPlotThemeManager:
    def test_plot_theme_construction(self) -> None:
        manager = PlotThemeManager()
        assert manager is not None

    def test_get_singleton(self) -> None:
        manager = get_plot_theme_manager()
        assert isinstance(manager, PlotThemeManager)

    def test_singleton_is_same(self) -> None:
        m1 = get_plot_theme_manager()
        m2 = get_plot_theme_manager()
        assert m1 is m2

    def test_get_current_theme_via_property(self) -> None:
        manager = PlotThemeManager()
        theme = manager.current_theme
        assert isinstance(theme, PlotTheme)
