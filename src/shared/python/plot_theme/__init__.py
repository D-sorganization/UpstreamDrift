"""Shared plot theming system for matplotlib and scientific visualizations.

This module provides a consistent color theme system for plots across all
D-sorganization applications, similar to the app theme system for PyQt6.

Usage:
    from plot_theme import PlotThemeManager, apply_plot_theme

    # Apply a theme to matplotlib
    manager = PlotThemeManager()
    manager.set_theme("scientific_violet")
    manager.apply_to_matplotlib()

    # Or use the convenience function
    apply_plot_theme("scientific_violet")
"""

from .integration import (
    PlotThemeMixin,
    apply_plot_theme,
    create_plot_theme_menu,
    create_themed_figure,
    get_theme_colors,
    setup_plot_theme_for_app,
    style_axis,
)
from .manager import PlotThemeManager, get_plot_theme_manager
from .themes import (
    PLOT_THEMES,
    PlotTheme,
    get_theme,
    get_theme_names,
    register_theme,
)

__all__ = [
    # Manager
    "PlotThemeManager",
    "get_plot_theme_manager",
    # Themes
    "PLOT_THEMES",
    "PlotTheme",
    "get_theme",
    "get_theme_names",
    "register_theme",
    # Integration
    "apply_plot_theme",
    "create_plot_theme_menu",
    "create_themed_figure",
    "get_theme_colors",
    "PlotThemeMixin",
    "setup_plot_theme_for_app",
    "style_axis",
]
