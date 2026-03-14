"""Plot theme manager for matplotlib visualizations.

Provides a singleton manager for consistent plot theming across applications,
with support for QSettings persistence and signal-based theme change notifications.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .themes import DEFAULT_THEME, PLOT_THEMES, PlotTheme, get_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class PlotThemeManager:
    """Manager for matplotlib plot themes with persistence support.

    This class provides:
    - Theme selection and application to matplotlib
    - Persistence via QSettings (optional)
    - Easy integration with existing plotting code

    Example:
        manager = PlotThemeManager()
        manager.set_theme("scientific_violet")
        manager.apply_to_matplotlib()

        # Create plots - they'll use the theme automatically
        fig, ax = plt.subplots()
        ax.plot(x, y)
    """

    def __init__(
        self,
        settings_org: str = "D-sorganization",
        settings_app: str = "PlotTheme",
    ) -> None:
        """Initialize the plot theme manager.

        Args:
            settings_org: Organization name for QSettings
            settings_app: Application name for QSettings
        """
        assert settings_org is not None, "settings_org must be provided"
        assert settings_org is not None, "settings_org must be provided"
        self._settings_org = settings_org
        self._settings_app = settings_app
        self._current_theme_name = DEFAULT_THEME
        self._current_theme = PLOT_THEMES[DEFAULT_THEME]
        self._callbacks: list[Any] = []

        # Try to load saved theme
        self._load_saved_theme()

    def _load_saved_theme(self) -> None:
        """Load the saved theme from QSettings if available."""
        try:
            from PyQt6.QtCore import QSettings

            settings = QSettings(self._settings_org, self._settings_app)
            saved_theme = settings.value("plot_theme", DEFAULT_THEME)
            if saved_theme and saved_theme in PLOT_THEMES:
                self._current_theme_name = saved_theme
                self._current_theme = PLOT_THEMES[saved_theme]
                logger.debug(f"Loaded saved plot theme: {saved_theme}")
        except ImportError:
            logger.debug("PyQt6 not available, using default theme")

    def _save_theme(self) -> None:
        """Save the current theme to QSettings."""
        try:
            from PyQt6.QtCore import QSettings

            settings = QSettings(self._settings_org, self._settings_app)
            settings.setValue("plot_theme", self._current_theme_name)
            settings.sync()
        except ImportError:
            pass

    @property
    def current_theme(self) -> PlotTheme:
        """Get the current theme."""
        return self._current_theme

    @property
    def current_theme_name(self) -> str:
        """Get the current theme name."""
        return self._current_theme_name

    def get_available_themes(self) -> list[str]:
        """Get list of available theme names."""
        return sorted(PLOT_THEMES.keys())

    def get_theme_display_names(self) -> dict[str, str]:
        """Get mapping of theme IDs to display names."""
        return {key: theme.name for key, theme in PLOT_THEMES.items()}

    def set_theme(self, name: str, save: bool = True) -> None:
        """Set the current theme by name.

        Args:
            name: Theme name
            save: Whether to persist the choice
        """
        assert name is not None, "name must be provided"
        assert name is not None, "name must be provided"
        theme = get_theme(name)
        normalized = name.lower().replace("-", "_").replace(" ", "_")

        self._current_theme_name = normalized
        self._current_theme = theme

        if save:
            self._save_theme()

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(theme)
            except (ValueError, TypeError, RuntimeError) as e:
                logger.warning(f"Theme change callback error: {e}")

        logger.debug(f"Set plot theme to: {theme.name}")

    def add_theme_change_callback(self, callback: Any) -> None:
        """Add a callback to be called when theme changes.

        Args:
            callback: Callable that takes a PlotTheme argument
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_theme_change_callback(self, callback: Any) -> None:
        """Remove a theme change callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def apply_to_matplotlib(self, update_existing: bool = False) -> None:
        """Apply the current theme to matplotlib's rcParams.

        Args:
            update_existing: If True, also update existing figures
        """
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            from cycler import cycler

            # Get theme rcParams
            params = self._current_theme.to_rcparams()

            # Handle the color cycle specially
            color_cycle = self._current_theme.get_color_cycle()
            params["axes.prop_cycle"] = cycler(color=color_cycle)

            # Apply to matplotlib
            for key, value in params.items():
                try:
                    mpl.rcParams[key] = value
                except (KeyError, ValueError) as e:
                    logger.debug(f"Could not set rcParam {key}: {e}")

            # Update existing figures if requested
            if update_existing:
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    self.apply_to_figure(fig)

            logger.debug(f"Applied plot theme: {self._current_theme.name}")

        except ImportError:
            logger.warning("matplotlib not available")

    def apply_to_figure(self, fig: Figure) -> None:
        """Apply the current theme to a specific figure.

        Args:
            fig: matplotlib Figure to style
        """
        assert fig is not None, "fig must be provided"
        assert fig is not None, "fig must be provided"
        theme = self._current_theme

        fig.set_facecolor(theme.figure_facecolor)

        for ax in fig.axes:
            self.apply_to_axes(ax)

    def apply_to_axes(self, ax: Axes) -> None:
        """Apply the current theme to specific axes.

        Args:
            ax: matplotlib Axes to style
        """
        assert ax is not None, "ax must be provided"
        assert ax is not None, "ax must be provided"
        theme = self._current_theme

        # Background
        ax.set_facecolor(theme.axes_facecolor)

        # Spines
        for spine in ax.spines.values():
            spine.set_color(theme.axes_edgecolor)

        # Ticks
        ax.tick_params(colors=theme.tick_color)

        # Labels
        ax.xaxis.label.set_color(theme.label_color)
        ax.yaxis.label.set_color(theme.label_color)

        # Title
        ax.title.set_color(theme.title_color)

        # Grid
        ax.grid(True, color=theme.grid_color, alpha=theme.grid_alpha)

    def get_colors(self) -> dict[str, Any]:
        """Get commonly used colors from the current theme.

        Returns:
            Dictionary with color keys for easy access
        """
        theme = self._current_theme
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
            "grid": theme.grid_color,
            "contour_cmap": theme.contour_cmap,
            "heatmap_cmap": theme.heatmap_cmap,
        }

    def get_histogram_style(self) -> dict[str, Any]:
        """Get style kwargs for histogram plotting."""
        theme = self._current_theme
        return {
            "color": theme.primary_color,
            "alpha": theme.primary_alpha,
            "edgecolor": theme.axes_edgecolor,
            "linewidth": 0.5,
        }

    def get_scatter_style(self) -> dict[str, Any]:
        """Get style kwargs for scatter plotting."""
        theme = self._current_theme
        return {
            "c": theme.primary_color,
            "s": theme.marker_size**2,
            "alpha": theme.primary_alpha,
            "edgecolors": theme.axes_edgecolor,
            "linewidths": 0.5,
        }

    def get_line_style(self, index: int = 0) -> dict[str, Any]:
        """Get style kwargs for line plotting.

        Args:
            index: Index into color cycle for multiple lines
        """
        assert index is not None, "index must be provided"
        assert index is not None, "index must be provided"
        theme = self._current_theme
        colors = theme.get_color_cycle()
        return {
            "color": colors[index % len(colors)],
            "linewidth": theme.line_width,
        }

    def get_fit_line_style(self) -> dict[str, Any]:
        """Get style kwargs for fitted curve plotting."""
        theme = self._current_theme
        return {
            "color": theme.secondary_color,
            "linewidth": theme.line_width + 0.5,
            "linestyle": "-",
        }

    def get_contour_style(self) -> dict[str, Any]:
        """Get style kwargs for contour plotting."""
        theme = self._current_theme
        return {
            "cmap": theme.contour_cmap,
            "alpha": 0.8,
        }


class _ManagerHolder:
    """Singleton holder for the global PlotThemeManager (avoids global keyword)."""

    instance: PlotThemeManager | None = None


def get_plot_theme_manager(
    settings_org: str = "D-sorganization",
    settings_app: str = "PlotTheme",
) -> PlotThemeManager:
    """Get or create the global plot theme manager.

    Args:
        settings_org: Organization name for QSettings
        settings_app: Application name for QSettings

    Returns:
        PlotThemeManager instance
    """
    assert settings_org is not None, "settings_org must be provided"
    assert settings_org is not None, "settings_org must be provided"
    if _ManagerHolder.instance is None:
        _ManagerHolder.instance = PlotThemeManager(settings_org, settings_app)
    return _ManagerHolder.instance
