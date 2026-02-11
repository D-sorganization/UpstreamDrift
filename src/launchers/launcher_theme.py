"""Theme management mixin for GolfLauncher.

Contains theme application, theme menu setup, plot theme management,
and dynamic theme change handling.
"""

from __future__ import annotations

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class LauncherThemeMixin:
    """Mixin for GolfLauncher theme management.

    Provides methods for applying styles, managing theme menus,
    and handling dynamic theme changes.
    """

    def apply_styles(self) -> None:
        """Apply themed stylesheet from the shared ThemeManager."""
        try:
            from src.shared.python.theme import ThemeManager

            manager = ThemeManager.instance()
            c = manager.colors
            self.setStyleSheet(manager.get_stylesheet() + f"""
                QScrollArea {{ border: none; }}
                QMenu::separator {{
                    height: 1px;
                    margin: 4px 8px;
                }}
                QFrame#ModelCard {{
                    background-color: {c.bg_elevated};
                    border: 1px solid {c.border_default};
                    border-radius: 12px;
                }}
                QFrame#ModelCard:hover {{
                    background-color: {c.bg_highlight};
                    border: 1px solid {c.border_strong};
                }}
                QLabel#CardDescription {{
                    color: {c.text_secondary};
                }}
            """)
        except ImportError:
            # Fallback minimal dark style if theme system unavailable
            self.setStyleSheet(
                "QMainWindow { background-color: #1E1E1E; }"
                "QWidget { color: #FFFFFF; font-family: 'Segoe UI', sans-serif; }"
            )

    def _apply_theme_system(self) -> None:
        """Initialize theme manager and register for theme change callbacks."""
        try:
            from src.shared.python.theme import ThemeManager, apply_golf_suite_style

            self._theme_manager = ThemeManager.instance()

            # Restore saved theme preference
            self._theme_manager.load_saved_theme()

            # Apply matplotlib styling globally
            apply_golf_suite_style()

            # Register callback for dynamic theme switching
            self._theme_manager.on_theme_changed(self._on_theme_changed)

        except ImportError as e:
            logger.warning(f"Theme system unavailable: {e}")

    def _on_theme_changed(self, colors: object) -> None:
        """Handle dynamic theme change -- reapply stylesheet and update menu."""
        self.apply_styles()

        # Refresh all model card inline styles
        for card in self.model_cards.values():
            if hasattr(card, "refresh_theme"):
                card.refresh_theme()

        # Reapply card selection state with new theme colors
        if self.selected_model:
            self.select_model(self.selected_model)
        else:
            self.update_launch_button()

        # Update the checked state of theme menu actions
        if hasattr(self, "_theme_actions"):
            from src.shared.python.theme import ThemeManager

            current = ThemeManager.instance().theme_name
            for action in self._theme_actions:
                action.setChecked(action.text() == current)

    def _setup_theme_menu(self, theme_menu: QMenu) -> None:
        """Populate the View > Theme submenu with all available themes.

        Includes core presets (Dark, Light, High Contrast), fleet-wide themes,
        custom themes, a "Manage Themes..." dialog, and a Plot Theme submenu.
        """
        from PyQt6.QtGui import QActionGroup

        try:
            from src.shared.python.theme import ThemeManager, ThemePreset

            manager = ThemeManager.instance()

            group = QActionGroup(self)
            group.setExclusive(True)
            self._theme_actions: list[QAction] = []

            # Core presets
            preset_map: dict[str, ThemePreset] = {
                "Dark": ThemePreset.DARK,
                "Light": ThemePreset.LIGHT,
                "High Contrast": ThemePreset.HIGH_CONTRAST,
            }
            for name, preset in preset_map.items():
                action = QAction(name, self)
                action.setCheckable(True)
                action.setChecked(manager.theme_name == name)
                action.triggered.connect(lambda checked, p=preset: manager.set_theme(p))
                group.addAction(action)
                theme_menu.addAction(action)
                self._theme_actions.append(action)

            # Fleet-wide themes
            fleet_names = manager.get_available_fleet_themes()
            if fleet_names:
                theme_menu.addSeparator()
                for fleet_name in fleet_names:
                    if fleet_name in preset_map:
                        continue
                    action = QAction(fleet_name, self)
                    action.setCheckable(True)
                    action.setChecked(manager.theme_name == fleet_name)
                    action.triggered.connect(
                        lambda checked, n=fleet_name: manager.set_fleet_theme(n)
                    )
                    group.addAction(action)
                    theme_menu.addAction(action)
                    self._theme_actions.append(action)

            # Custom themes
            custom_names = manager.get_custom_theme_names()
            if custom_names:
                theme_menu.addSeparator()
                for cname in custom_names:
                    action = QAction(cname, self)
                    action.setCheckable(True)
                    action.setChecked(manager.theme_name == cname)
                    action.triggered.connect(
                        lambda checked, n=cname: manager.change_theme(n)
                    )
                    group.addAction(action)
                    theme_menu.addAction(action)
                    self._theme_actions.append(action)

            # Manage Themes dialog
            theme_menu.addSeparator()
            manage_action = QAction("Manage Themes...", self)
            manage_action.triggered.connect(self._open_theme_manager_dialog)
            theme_menu.addAction(manage_action)

            # Plot Theme submenu
            theme_menu.addSeparator()
            plot_menu = theme_menu.addMenu("Plot Theme")
            if plot_menu:
                self._setup_plot_theme_menu(plot_menu)

        except ImportError as e:
            logger.warning(f"Could not populate theme menu: {e}")
            fallback = QAction("(Theme system unavailable)", self)
            fallback.setEnabled(False)
            theme_menu.addAction(fallback)

    def _open_theme_manager_dialog(self) -> None:
        """Open the full Theme Manager dialog."""
        try:
            from src.shared.python.theme import ThemeManager
            from src.shared.python.theme.dialogs import ThemeManagerDialog

            manager = ThemeManager.instance()
            dialog = ThemeManagerDialog(manager, self)
            dialog.theme_changed.connect(lambda _: self._on_theme_changed(None))
            dialog.exec()
        except ImportError as e:
            logger.error(f"Could not open Theme Manager: {e}")

    def _setup_plot_theme_menu(self, plot_menu: QMenu) -> None:
        """Populate the Plot Theme submenu.

        Plot themes affect matplotlib styling used by submodules.
        The setting is saved to QSettings so launched modules inherit it.
        """
        from PyQt6.QtCore import QSettings
        from PyQt6.QtGui import QActionGroup

        group = QActionGroup(self)
        group.setExclusive(True)

        settings = QSettings("UpstreamDrift", "GolfModelingSuite")
        current_plot = settings.value("plot_theme", "follow_ui")

        # "Follow UI Theme" option
        follow_action = QAction("Follow UI Theme (Recommended)", self)
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
                action = QAction(style_name, self)
                action.setCheckable(True)
                action.setChecked(current_plot == style_name)
                action.triggered.connect(
                    lambda checked, s=style_name: self._set_plot_theme(s)
                )
                group.addAction(action)
                plot_menu.addAction(action)
        except ImportError:
            na = QAction("(matplotlib not available)", self)
            na.setEnabled(False)
            plot_menu.addAction(na)

    def _set_plot_theme(self, theme_name: str) -> None:
        """Save plot theme preference to QSettings."""
        from PyQt6.QtCore import QSettings

        settings = QSettings("UpstreamDrift", "GolfModelingSuite")
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
