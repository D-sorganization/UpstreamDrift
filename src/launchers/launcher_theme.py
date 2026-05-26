"""Theme management mixin for UpstreamDriftLauncher.

Contains theme application, theme menu setup, plot theme management,
and dynamic theme change handling.
"""

# mypy: disable-error-code="attr-defined,call-overload,arg-type"

from __future__ import annotations

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.typography import CSS_FONT_UI

logger = get_logger(__name__)


class ThemeManager:
    def __init__(self, launcher):
        self.launcher = launcher

    def __getattr__(self, name):
        return getattr(self.launcher, name)

    def __setattr__(self, name, value):
        if name == "launcher" or hasattr(type(self), name) or name in self.__dict__:
            super().__setattr__(name, value)
        elif hasattr(self.launcher, name):
            setattr(self.launcher, name, value)
        else:
            super().__setattr__(name, value)

    """Mixin for UpstreamDriftLauncher theme management.

    Provides methods for applying styles, managing theme menus,
    and handling dynamic theme changes.
    """

    def apply_styles(self) -> None:
        """Apply themed stylesheet from the shared ThemeManager."""
        try:
            from src.shared.python.theme import ThemeManager

            manager = ThemeManager.instance()
            colors = manager.get_current_colors()

            # Map extended color names to actual theme keys with fallbacks
            bg_elevated = colors.get("bg_elevated", colors.get("group_bg", "#2D2D2D"))
            border_default = colors.get(  # noqa: F841
                "border_default", colors.get("border", "#555555")
            )
            bg_highlight = colors.get("bg_highlight", colors.get("input_bg", "#3D3D3D"))  # noqa: F841
            border_strong = colors.get("border_strong", colors.get("focus", "#0078D4"))  # noqa: F841
            text_sec = colors.get("text_secondary", "#AAAAAA")

            self.setStyleSheet(
                manager.get_current_stylesheet()
                + f"""
                QMainWindow {{ background-color: {bg_elevated}; }}
                QScrollArea {{ border: none; }}
                QMenu::separator {{
                    height: 1px;
                    margin: 4px 8px;
                }}
                QLabel#CardDescription {{
                    color: {text_sec};
                }}
            """
            )
        except (ImportError, AttributeError):
            # Fallback minimal dark style if theme system unavailable
            self.setStyleSheet(
                "QMainWindow { background-color: #1E1E1E; }"
                f"QWidget {{ color: #FFFFFF; {CSS_FONT_UI} }}"
            )

    def _apply_theme_system(self) -> None:
        """Initialize theme manager and register for theme change callbacks."""
        try:
            from src.shared.python.theme import (
                ThemeManager,
                FontManager,
                apply_golf_suite_style,
            )

            self._theme_manager = ThemeManager.instance()

            # Initialize FontManager
            if FontManager is not None:
                self._font_manager = FontManager(
                    app_context="UpstreamDrift",
                    settings_org="D-sorganization",
                    settings_app="UpstreamDrift",
                )
                self._font_manager.apply_font()

            # Apply matplotlib styling globally
            apply_golf_suite_style()

            # Register callback for dynamic theme switching via Qt signal
            if hasattr(self._theme_manager, "themeChanged"):
                self._theme_manager.themeChanged.connect(self._on_theme_changed)

        except ImportError as e:
            logger.warning(f"Theme system unavailable: {e}")

    def _on_theme_changed(self, colors: object = None) -> None:
        """Handle dynamic theme change -- reapply stylesheet and update menu."""
        self.apply_styles()

        # Refresh all model card inline styles
        for card in self.model_cards.values():
            if hasattr(card, "refresh_theme"):
                card.refresh_theme()

        # Refresh AI panel if it exists
        if hasattr(self, "ai_panel") and hasattr(self.ai_panel, "refresh_theme"):
            self.ai_panel.refresh_theme()

        # Reapply card selection state with new theme colors
        if self.selected_model:
            self.select_model(self.selected_model)
        else:
            self.update_launch_button()

        # Update the checked state of theme menu actions
        if hasattr(self, "_theme_actions"):
            from src.shared.python.theme import ThemeManager

            current = ThemeManager.instance().get_current_theme_name()
            for action in self._theme_actions:
                action.setChecked(action.text() == current)

    def _setup_theme_menu(self, theme_menu: QMenu) -> None:
        """Populate the View > Theme submenu with all available themes.

        Includes core presets (Dark, Light, High Contrast), fleet-wide themes,
        custom themes, a "Manage Themes..." dialog, and a Plot Theme submenu.
        """
        if theme_menu is None:
            raise ValueError("theme_menu must be provided")
        from PyQt6.QtGui import QActionGroup

        try:
            from src.shared.python.theme import ThemeManager, ThemePreset

            manager = ThemeManager.instance()

            group = QActionGroup(self.launcher)
            group.setExclusive(True)
            self._theme_actions: list[QAction] = []

            # Core presets
            preset_map: dict[str, ThemePreset] = {
                "Dark": ThemePreset.DARK,
                "Light": ThemePreset.LIGHT,
                "High Contrast": ThemePreset.HIGH_CONTRAST,
            }
            for name, preset in preset_map.items():
                action = QAction(name, self.launcher)
                action.setCheckable(True)
                action.setChecked(manager.get_current_theme_name() == name)
                action.triggered.connect(
                    lambda checked, p=preset: manager.change_theme(p.value)
                )
                group.addAction(action)
                theme_menu.addAction(action)
                self._theme_actions.append(action)

            # Additional built-in themes beyond the core presets
            all_themes = manager.get_available_themes()
            extra_themes = [t for t in all_themes if t not in preset_map]
            if extra_themes:
                theme_menu.addSeparator()
                for theme_name in extra_themes:
                    action = QAction(theme_name, self.launcher)
                    action.setCheckable(True)
                    action.setChecked(manager.get_current_theme_name() == theme_name)
                    action.triggered.connect(
                        lambda checked, n=theme_name: manager.change_theme(n)
                    )
                    group.addAction(action)
                    theme_menu.addAction(action)
                    self._theme_actions.append(action)

            # Custom themes
            custom_names = manager.get_custom_theme_names()
            if custom_names:
                theme_menu.addSeparator()
                for cname in custom_names:
                    action = QAction(cname, self.launcher)
                    action.setCheckable(True)
                    action.setChecked(manager.get_current_theme_name() == cname)
                    action.triggered.connect(
                        lambda checked, n=cname: manager.change_theme(n)
                    )
                    group.addAction(action)
                    theme_menu.addAction(action)
                    self._theme_actions.append(action)

            # Manage Themes dialog
            theme_menu.addSeparator()
            manage_action = QAction("Manage Themes...", self.launcher)
            manage_action.triggered.connect(self._open_theme_manager_dialog)
            theme_menu.addAction(manage_action)

            # Typography submenu
            theme_menu.addSeparator()
            typography_menu = theme_menu.addMenu("Typography")
            if typography_menu:
                self._setup_typography_menu(typography_menu)

            # Plot Theme submenu
            theme_menu.addSeparator()
            plot_menu = theme_menu.addMenu("Plot Theme")
            if plot_menu:
                self._setup_plot_theme_menu(plot_menu)

        except ImportError as e:
            logger.warning(f"Could not populate theme menu: {e}")
            fallback = QAction("(Theme system unavailable)", self.launcher)
            fallback.setEnabled(False)
            theme_menu.addAction(fallback)

    def _setup_typography_menu(self, typography_menu: QMenu) -> None:
        """Populate the Typography submenu."""
        if typography_menu is None:
            raise ValueError("typography_menu must be provided")
        from PyQt6.QtGui import QActionGroup

        try:
            from src.shared.python.theme.font_manager import (
                FontManager,
                get_font_manager,
            )

            if FontManager is None:
                return

            manager = get_font_manager()
            group = QActionGroup(self.launcher)
            group.setExclusive(True)

            available_fonts = manager.get_available_fonts()
            current_font = manager.get_current_font()

            for font_name in available_fonts:
                action = QAction(font_name, self.launcher)
                action.setCheckable(True)
                action.setChecked(font_name == current_font)
                action.triggered.connect(
                    lambda checked, f=font_name: manager.change_font(f)
                )
                group.addAction(action)
                typography_menu.addAction(action)

        except ImportError as e:
            logger.warning(f"Could not populate typography menu: {e}")
            fallback = QAction("(Theme system unavailable)", self.launcher)
            fallback.setEnabled(False)
            typography_menu.addAction(fallback)

    def _open_theme_manager_dialog(self) -> None:
        """Open the full Theme Manager dialog."""
        try:
            from src.shared.python.theme import ThemeManager
            from src.shared.python.theme.dialogs import ThemeManagerDialog

            manager = ThemeManager.instance()
            dialog = ThemeManagerDialog(manager, self.launcher)
            dialog.theme_changed.connect(lambda _: self._on_theme_changed(None))
            dialog.exec()
        except ImportError as e:
            logger.error(f"Could not open Theme Manager: {e}")

    def _setup_plot_theme_menu(self, plot_menu: QMenu) -> None:
        """Populate the Plot Theme submenu.

        Plot themes affect matplotlib styling used by submodules.
        The setting is saved to QSettings so launched modules inherit it.
        """
        if plot_menu is None:
            raise ValueError("plot_menu must be provided")
        from PyQt6.QtCore import QSettings
        from PyQt6.QtGui import QActionGroup

        group = QActionGroup(self.launcher)
        group.setExclusive(True)

        settings = QSettings("UpstreamDrift", "Launcher")
        current_plot = settings.value("plot_theme", "follow_ui")

        # "Follow UI Theme" option
        follow_action = QAction("Follow UI Theme (Recommended)", self.launcher)
        follow_action.setCheckable(True)
        follow_action.setChecked(current_plot == "follow_ui")
        follow_action.triggered.connect(lambda: self._set_plot_theme("follow_ui"))
        group.addAction(follow_action)
        plot_menu.addAction(follow_action)

        plot_menu.addSeparator()

        # Matplotlib built-in styles
        try:
            import matplotlib.pyplot as plt

            for style_name in sorted(plt.style.available):
                if style_name.startswith("_"):
                    continue
                action = QAction(style_name, self.launcher)
                action.setCheckable(True)
                action.setChecked(current_plot == style_name)
                action.triggered.connect(
                    lambda checked, s=style_name: self._set_plot_theme(s)
                )
                group.addAction(action)
                plot_menu.addAction(action)
        except ImportError:
            na = QAction("(matplotlib not available)", self.launcher)
            na.setEnabled(False)
            plot_menu.addAction(na)

    def _set_plot_theme(self, theme_name: str) -> None:
        """Save plot theme preference to QSettings."""
        if theme_name is None:
            raise ValueError("theme_name must be provided")
        from PyQt6.QtCore import QSettings

        settings = QSettings("UpstreamDrift", "Launcher")
        settings.setValue("plot_theme", theme_name)
        logger.info("Plot theme set to: %s", theme_name)

        # Apply immediately if matplotlib is available
        if theme_name == "follow_ui":
            try:
                from src.shared.python.theme import apply_golf_suite_style

                apply_golf_suite_style()
            except ImportError:
                pass
        else:
            try:
                import matplotlib.pyplot as plt

                plt.style.use(theme_name)
            except ImportError:
                pass
