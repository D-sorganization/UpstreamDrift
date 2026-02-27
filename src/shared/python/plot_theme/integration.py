"""Integration helpers for plot theming.

This module provides easy-to-use functions and mixins for integrating
the plot theme system into applications.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .manager import PlotThemeManager, get_plot_theme_manager
from .themes import PlotTheme, get_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def apply_plot_theme(
    theme_name: str | None = None,
    settings_app: str = "PlotTheme",
) -> PlotThemeManager:
    """Apply a plot theme to matplotlib globally.

    This is the simplest way to use the plot theme system:

        from plot_theme import apply_plot_theme

        apply_plot_theme("scientific_violet")

        # All subsequent plots will use this theme
        plt.plot(x, y)

    Args:
        theme_name: Theme to apply (None = use saved/default)
        settings_app: App name for persistence

    Returns:
        PlotThemeManager instance for further customization
    """
    manager = get_plot_theme_manager(settings_app=settings_app)

    if theme_name:
        manager.set_theme(theme_name)

    manager.apply_to_matplotlib()
    return manager


def create_themed_figure(
    figsize: tuple[float, float] = (10, 6),
    theme_name: str | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a new figure with theme applied.

    Args:
        figsize: Figure size in inches
        theme_name: Theme to use (None = current theme)
        **kwargs: Additional arguments for plt.subplots()

    Returns:
        Tuple of (Figure, Axes)
    """
    import matplotlib.pyplot as plt

    manager = get_plot_theme_manager()

    if theme_name:
        manager.set_theme(theme_name, save=False)
        manager.apply_to_matplotlib()

    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    manager.apply_to_figure(fig)

    return fig, ax


def style_axis(
    ax: Axes,
    theme_name: str | None = None,
) -> None:
    """Apply theme styling to existing axes.

    Args:
        ax: Axes to style
        theme_name: Theme to use (None = current theme)
    """
    manager = get_plot_theme_manager()

    theme = get_theme(theme_name) if theme_name else manager.current_theme

    # Apply styling
    ax.set_facecolor(theme.axes_facecolor)

    for spine in ax.spines.values():
        spine.set_color(theme.axes_edgecolor)

    ax.tick_params(colors=theme.tick_color)
    ax.xaxis.label.set_color(theme.label_color)
    ax.yaxis.label.set_color(theme.label_color)
    ax.title.set_color(theme.title_color)
    ax.grid(True, color=theme.grid_color, alpha=theme.grid_alpha)


def get_theme_colors(theme_name: str | None = None) -> dict[str, Any]:
    """Get color dictionary from a theme.

    Args:
        theme_name: Theme name (None = current theme)

    Returns:
        Dictionary with color values
    """
    manager = get_plot_theme_manager()

    theme = get_theme(theme_name) if theme_name else manager.current_theme

    return {
        "primary": theme.primary_color,
        "primary_colors": theme.primary_colors,
        "secondary": theme.secondary_color,
        "secondary_colors": theme.secondary_colors,
        "accent": theme.accent_color,
        "accent_colors": theme.accent_colors,
        "background": theme.axes_facecolor,
        "figure_background": theme.figure_facecolor,
        "text": theme.text_color,
        "title": theme.title_color,
        "label": theme.label_color,
        "grid": theme.grid_color,
        "edge": theme.axes_edgecolor,
        "contour_cmap": theme.contour_cmap,
        "heatmap_cmap": theme.heatmap_cmap,
    }


class PlotThemeMixin:
    """Mixin class for adding plot theme support to widgets/windows.

    Usage:
        class MyPlotWindow(QMainWindow, PlotThemeMixin):
            def __init__(self):
                super().__init__()
                self.setup_plot_theme(settings_app="MyApp")

            def on_plot_theme_changed(self, theme):
                self.update_plots()
    """

    _plot_theme_manager: PlotThemeManager | None = None

    def setup_plot_theme(
        self,
        settings_org: str = "D-sorganization",
        settings_app: str = "PlotTheme",
        apply_immediately: bool = True,
    ) -> PlotThemeManager:
        """Set up plot theme support.

        Args:
            settings_org: Organization name for QSettings
            settings_app: Application name for QSettings
            apply_immediately: Apply theme to matplotlib now

        Returns:
            PlotThemeManager instance
        """
        self._plot_theme_manager = get_plot_theme_manager(
            settings_org=settings_org,
            settings_app=settings_app,
        )

        # Add callback for theme changes
        self._plot_theme_manager.add_theme_change_callback(
            self._on_plot_theme_changed_internal
        )

        if apply_immediately:
            self._plot_theme_manager.apply_to_matplotlib()

        return self._plot_theme_manager

    def _on_plot_theme_changed_internal(self, theme: PlotTheme) -> None:
        """Internal handler for theme changes."""
        # Apply to matplotlib
        if self._plot_theme_manager:
            self._plot_theme_manager.apply_to_matplotlib()

        # Call subclass handler if defined
        if hasattr(self, "on_plot_theme_changed"):
            self.on_plot_theme_changed(theme)

    def get_plot_theme_manager(self) -> PlotThemeManager | None:
        """Get the plot theme manager."""
        return self._plot_theme_manager

    def set_plot_theme(self, name: str) -> None:
        """Set the plot theme by name.

        Args:
            name: Theme name
        """
        if self._plot_theme_manager:
            self._plot_theme_manager.set_theme(name)

    def get_plot_colors(self) -> dict[str, Any]:
        """Get current theme colors."""
        if self._plot_theme_manager:
            return self._plot_theme_manager.get_colors()
        return {}


def create_plot_theme_menu(
    parent: Any,
    menubar: Any | None = None,
) -> Any:
    """Create a Plot Theme menu for PyQt6 applications.

    Args:
        parent: Parent widget (usually QMainWindow)
        menubar: Menu bar to add to (None = get from parent)

    Returns:
        QMenu instance
    """
    try:
        from PyQt6.QtGui import QAction, QActionGroup
        from PyQt6.QtWidgets import QMenu
    except ImportError:
        logger.warning("PyQt6 not available for plot theme menu")
        return None

    manager = get_plot_theme_manager()

    # Get or create menu bar
    if menubar is None:
        menubar = parent.menuBar()

    # Create Plot Theme menu
    menu = QMenu("Plot Theme", parent)

    # Create action group for exclusive selection
    action_group = QActionGroup(parent)
    action_group.setExclusive(True)

    # Get theme display names
    theme_names = manager.get_theme_display_names()
    current = manager.current_theme_name

    # Add theme actions
    for theme_id, display_name in sorted(theme_names.items(), key=lambda x: x[1]):
        action = QAction(display_name, parent)
        action.setCheckable(True)
        action.setChecked(theme_id == current)

        # Create closure to capture theme_id
        def make_handler(tid: str) -> Any:
            def handler(checked: bool) -> None:
                if checked:
                    manager.set_theme(tid)
                    manager.apply_to_matplotlib(update_existing=True)

            return handler

        action.triggered.connect(make_handler(theme_id))
        action_group.addAction(action)
        menu.addAction(action)

    # Add to menubar
    menubar.addMenu(menu)

    return menu


def setup_plot_theme_for_app(
    app: Any,
    window: Any,
    settings_app: str | None = None,
    add_menu: bool = True,
) -> PlotThemeManager:
    """Set up plot theming for a PyQt6 application.

    This is a convenience function similar to setup_themed_app for UI themes.

    Args:
        app: QApplication instance
        window: QMainWindow instance
        settings_app: App name for persistence (None = use window class name)
        add_menu: Whether to add Plot Theme menu

    Returns:
        PlotThemeManager instance
    """
    if settings_app is None:
        settings_app = window.__class__.__name__

    manager = get_plot_theme_manager(settings_app=settings_app)
    manager.apply_to_matplotlib()

    if add_menu:
        create_plot_theme_menu(window)

    return manager
