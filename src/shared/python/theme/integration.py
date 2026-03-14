"""Theme integration helpers for PyQt6 applications.

This module provides easy-to-use functions and mixins for integrating
the fleet-wide theme system into PyQt6 applications.

Usage:
    # Simple function-based approach
    from shared.python.theme.integration import setup_themed_app

    app = QApplication(sys.argv)
    window = MyMainWindow()
    setup_themed_app(app, window)  # Applies theme and adds menu
    window.show()
    sys.exit(app.exec())

    # Mixin approach for more control
    from shared.python.theme.integration import ThemedWindowMixin

    class MyMainWindow(ThemedWindowMixin, QMainWindow):
        def __init__(self):
            super().__init__()
            self.setup_theme_support()  # Adds theme menu and applies theme
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PyQt6.QtWidgets import QApplication, QMenu, QMenuBar

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)


def get_theme_manager(
    window: QMainWindow | None = None,
    settings_org: str = "D-sorganization",
    settings_app: str = "FleetTheme",
) -> Any:
    """Get or create the singleton ThemeManager.

    Args:
        window: Optional main window to register with the manager
        settings_org: QSettings organization name
        settings_app: QSettings application name

    Returns:
        ThemeManager instance
    """
    assert settings_org is not None, "settings_org must be provided"
    from .theme_manager import ThemeManager

    return ThemeManager.instance(
        main_window=window,
        settings_org=settings_org,
        settings_app=settings_app,
    )


def apply_theme_to_window(window: QMainWindow, theme_name: str | None = None) -> None:
    """Apply theme to a window.

    If theme_name is None, applies the current theme from settings.

    Args:
        window: Window to apply theme to
        theme_name: Optional specific theme name, or None for current theme
    """
    assert window is not None, "window must be provided"
    manager = get_theme_manager(window)

    if theme_name:
        manager.change_theme(theme_name)
    else:
        manager.apply_theme()


def create_theme_menu(
    window: QMainWindow,
    parent_menu: QMenuBar | QMenu | None = None,
    show_custom_options: bool = False,
) -> QMenu:
    """Create a Theme menu with all available themes.

    Args:
        window: Main window to apply themes to
        parent_menu: Parent menubar or menu to add to
        show_custom_options: If True, add "Create Custom Theme..." and
            "Manage Themes..." actions at the bottom of the menu

    Returns:
        The created QMenu
    """
    assert window is not None, "window must be provided"
    from PyQt6.QtGui import QAction, QActionGroup

    manager = get_theme_manager(window)

    # Create the theme menu
    theme_menu = QMenu("&Theme", window)

    # Create action group for mutually exclusive selection
    theme_group = QActionGroup(window)
    theme_group.setExclusive(True)

    # Add all available themes
    current_theme = manager.get_current_theme_name()
    for theme_name in manager.get_available_themes():
        action = QAction(theme_name, window)
        action.setCheckable(True)
        action.setChecked(theme_name == current_theme)
        action.setData(theme_name)

        # Connect to theme change
        action.triggered.connect(
            lambda checked, name=theme_name: manager.change_theme(name)
        )

        theme_group.addAction(action)
        theme_menu.addAction(action)

    # Connect to theme changed signal to update checkmarks
    def update_checkmarks(new_theme: str) -> None:
        for action in theme_group.actions():
            action.setChecked(action.data() == new_theme)

    manager.themeChanged.connect(update_checkmarks)

    # Add custom theme options if requested
    if show_custom_options:
        theme_menu.addSeparator()

        create_action = QAction("Create Custom Theme...", window)
        create_action.triggered.connect(
            lambda: _open_custom_theme_editor(manager, window)
        )
        theme_menu.addAction(create_action)

        manage_action = QAction("Manage Themes...", window)
        manage_action.triggered.connect(
            lambda: _open_theme_manager_dialog(manager, window)
        )
        theme_menu.addAction(manage_action)

    # Add to parent if provided
    if parent_menu is not None:
        if isinstance(parent_menu, QMenuBar):
            parent_menu.addMenu(theme_menu)
        else:
            parent_menu.addMenu(theme_menu)

    return theme_menu


def _open_custom_theme_editor(manager: Any, window: QMainWindow) -> None:
    """Open the custom theme editor dialog."""
    assert window is not None, "window must be provided"
    from .dialogs import CustomThemeEditor

    editor = CustomThemeEditor(manager, window)
    editor.exec()


def _open_theme_manager_dialog(manager: Any, window: QMainWindow) -> None:
    """Open the theme manager dialog."""
    assert window is not None, "window must be provided"
    from .dialogs import ThemeManagerDialog

    dialog = ThemeManagerDialog(manager, window)
    dialog.exec()


def setup_themed_app(
    app: QApplication,
    window: QMainWindow,
    add_menu: bool = True,
    show_custom_options: bool = False,
    settings_org: str = "D-sorganization",
    settings_app: str | None = None,
) -> None:
    """Set up theme support for an application.

    This is the simplest way to add theme support:
    1. Initializes the ThemeManager
    2. Applies the saved theme (or default)
    3. Optionally adds a Theme menu to the menubar

    Args:
        app: QApplication instance
        window: Main window to theme
        add_menu: Whether to add a Theme menu to the menubar
        show_custom_options: If True, include custom theme create/manage actions
        settings_org: QSettings organization name
        settings_app: QSettings application name (defaults to window class name)
    """
    # Use window class name as default app name
    assert app is not None, "app must be provided"
    if settings_app is None:
        settings_app = window.__class__.__name__

    # Initialize theme manager
    manager = get_theme_manager(window, settings_org, settings_app)

    # Apply the current theme
    manager.apply_theme()

    # Add theme menu if requested and window has a menubar
    if add_menu:
        menubar = window.menuBar()
        if menubar is not None:
            create_theme_menu(window, menubar, show_custom_options=show_custom_options)

    logger.info(
        "Theme support initialized: theme=%s, app=%s",
        manager.get_current_theme_name(),
        settings_app,
    )


class ThemedWindowMixin:
    """Mixin class for adding theme support to QMainWindow subclasses.

    Usage:
        class MyMainWindow(ThemedWindowMixin, QMainWindow):
            def __init__(self):
                super().__init__()
                self.setup_theme_support()

    This mixin provides:
    - Automatic theme application on initialization
    - Theme menu in the menubar
    - Theme persistence via QSettings
    - Theme change notifications
    """

    _theme_manager = None
    _settings_org = "D-sorganization"
    _settings_app = None

    def setup_theme_support(
        self,
        add_menu: bool = True,
        show_custom_options: bool = False,
        settings_org: str | None = None,
        settings_app: str | None = None,
    ) -> None:
        """Initialize theme support for this window.

        Args:
            add_menu: Whether to add a Theme menu
            show_custom_options: If True, include custom theme create/manage actions
            settings_org: Override default settings organization
            settings_app: Override default settings application name
        """
        assert add_menu is not None, "add_menu must be provided"
        if settings_org:
            self._settings_org = settings_org
        if settings_app:
            self._settings_app = settings_app
        elif self._settings_app is None:
            self._settings_app = self.__class__.__name__

        # Get theme manager (self is expected to be a QMainWindow)
        self._theme_manager = get_theme_manager(
            self,  # type: ignore[arg-type]
            self._settings_org,
            self._settings_app,
        )

        # Apply current theme
        self._theme_manager.apply_theme()

        # Add theme menu
        if add_menu:
            menubar = self.menuBar()  # type: ignore[attr-defined]
            if menubar is not None:
                create_theme_menu(
                    self,  # type: ignore[arg-type]
                    menubar,
                    show_custom_options=show_custom_options,
                )

        # Connect to theme changes for custom handling
        self._theme_manager.themeChanged.connect(self._on_theme_changed)

        logger.info(
            "Theme support initialized for %s: theme=%s",
            self.__class__.__name__,
            self._theme_manager.get_current_theme_name(),
        )

    def _on_theme_changed(self, theme_name: str) -> None:
        """Called when the theme changes.

        Override this method to perform custom actions on theme change.
        The stylesheet is automatically applied before this is called.

        Args:
            theme_name: Name of the new theme
        """

    def get_theme_manager(self) -> Any:
        """Get the theme manager instance."""
        return self._theme_manager

    def change_theme(self, theme_name: str) -> None:
        """Change to a specific theme.

        Args:
            theme_name: Name of the theme to apply
        """
        if self._theme_manager:
            self._theme_manager.change_theme(theme_name)

    def get_current_theme(self) -> str:
        """Get the current theme name."""
        if self._theme_manager:
            return str(self._theme_manager.get_current_theme_name())
        return "Light"


__all__ = [
    "ThemedWindowMixin",
    "apply_theme_to_window",
    "create_theme_menu",
    "get_theme_manager",
    "setup_themed_app",
]
