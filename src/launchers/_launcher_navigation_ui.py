"""Launcher navigation, sidebar, and menu construction (#8490)."""

# mypy: disable-error-code="attr-defined,call-overload,arg-type,assignment"

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QMenuBar,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.launchers.custom_title_bar import create_window_control_button
from src.launchers.launcher_constants import ViewMode
from src.shared.python.theme.style_constants import Styles


def _build_menu_bar_close_widget(
    parent: QWidget,
    close_callback: Any,
    *,
    button_factory: Any = create_window_control_button,
) -> QWidget:
    """Create the top-row close control for the launcher menu bar."""
    container = QWidget(parent)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 1, 6, 1)
    layout.setSpacing(0)

    close_button = button_factory(
        "close",
        "X",
        tooltip="Close the launcher",
        accessible_name="Close launcher window",
        object_name="menu-bar-close-button",
        parent=container,
    )
    close_button.clicked.connect(lambda _checked=False: close_callback())
    layout.addWidget(close_button)
    return container


class LauncherNavigationUIMixin:
    """Build the launcher's global navigation and application menus."""

    def _build_sidebar_button(
        self,
        label: str,
        icon_name: str,
        *,
        checkable: bool = False,
    ) -> QToolButton:
        """Create an icon-first sidebar control with accessible labeling.

        Provides both icon and visible text label for accessibility.
        """
        button = QToolButton()
        button.setText(label)
        button.setToolTip(label)
        button.setAccessibleName(label)
        button.setAccessibleDescription(f"Navigate to {label} section")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        # Show both icon and text for better accessibility
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        # Sidebar buttons must always render a visible icon (#5624).
        # If the requested glyph is missing from SVG_REGISTRY, fall back
        # to a registered glyph so the icon is never null — null icons
        # produce the text-only sidebar regression visible in #5624.
        try:
            from src.shared.python.theme.icon_utils import (
                SVG_REGISTRY,
                IconColorizer,
            )

            resolved_name = icon_name if icon_name in SVG_REGISTRY else "settings"
            button.setIcon(IconColorizer.get_icon(resolved_name, "#d4d4d4"))
        except (ImportError, ValueError):
            # Last-resort fallback: any QIcon (even with a tinted blank
            # pixmap) is preferable to a null icon for hit-testing and
            # screen-reader semantics.
            from PyQt6.QtGui import QIcon, QPixmap

            pixmap = QPixmap(22, 22)
            pixmap.fill(Qt.GlobalColor.transparent)
            button.setIcon(QIcon(pixmap))

        button.setIconSize(QSize(22, 22))
        button.setCheckable(checkable)
        button.setAutoRaise(True)
        # Set focus policy for keyboard navigation
        button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return button

    def _setup_global_sidebar(self) -> QWidget:
        """Create the global sidebar navigation."""
        sidebar, layout, colors = self._create_sidebar_shell()
        filter_buttons = self._build_sidebar_filter_buttons()
        btn_settings, btn_console = self._build_sidebar_utility_buttons()
        btn_library = self._register_sidebar_buttons(filter_buttons)
        ordered_buttons = [
            *(button for button, _button_id in filter_buttons[:8]),
            btn_library,
            *(button for button, _button_id in filter_buttons[8:]),
        ]
        self._populate_sidebar_layout(
            layout,
            ordered_buttons,
            btn_console=btn_console,
            btn_settings=btn_settings,
        )
        self._set_sidebar_focus_order(
            sidebar,
            [*ordered_buttons, btn_console, btn_settings],
        )
        return self._wrap_sidebar(sidebar, colors)

    def _create_sidebar_shell(self) -> tuple[QWidget, QVBoxLayout, Any]:
        """Create and style the sidebar content widget."""
        sidebar = QWidget()
        sidebar.setMinimumWidth(Styles.SIDEBAR_MIN_WIDTH)
        sidebar.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )

        try:
            from src.shared.python.theme import get_current_colors

            colors = get_current_colors()
        except ImportError:
            from src.shared.python.theme import DARK_THEME

            colors = DARK_THEME

        sidebar.setStyleSheet(f"""
            QWidget {{
                background-color: {colors.bg_elevated};
                border-right: 1px solid {colors.border_default};
            }}
            QToolButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                color: {colors.text_secondary};
                padding: 12px 0;
            }}
            QToolButton:hover {{
                background-color: {colors.bg_highlight};
                color: {colors.text_primary};
                border: 1px solid {colors.border_default};
            }}
            QToolButton:checked {{
                background-color: {colors.primary};
                color: {colors.bg};
                border: 1px solid {colors.primary};
            }}
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 20, 8, 20)
        layout.setSpacing(15)
        return sidebar, layout, colors

    def _build_sidebar_filter_buttons(self) -> list[tuple[QToolButton, int]]:
        """Build category buttons in their stable routing order."""
        specs = (
            ("Home", "home", None),
            ("Engines", "computer", None),
            (
                "Biomechanics",
                "accessibility",
                "Filter tiles to show biomechanics and motion analysis tools",
            ),
            ("Simulation", "sports_golf", None),
            (
                "Motion Match",
                "directions_run",
                "Filter tiles to show motion matching tools",
            ),
            ("MoCap", "videocam", "Filter tiles to show motion capture tools"),
            ("Tools", "build", None),
            (
                "Documentation",
                "book",
                "Filter tiles to show documentation and library tools",
            ),
            ("Favorites", "star", "Filter tiles to show your favorites"),
            (
                "History",
                "history",
                "Filter tiles to show recently and frequently used tools",
            ),
        )
        buttons: list[tuple[QToolButton, int]] = []
        for button_id, (label, icon_name, description) in enumerate(specs):
            button = self._build_sidebar_button(label, icon_name, checkable=True)
            if description is not None:
                button.setAccessibleDescription(description)
            buttons.append((button, button_id))
        buttons[0][0].setChecked(True)
        return buttons

    def _build_sidebar_utility_buttons(self) -> tuple[QToolButton, QToolButton]:
        """Build the non-filtering Settings and Console controls."""
        btn_settings = self._build_sidebar_button(
            "Settings",
            "settings",
            checkable=False,
        )
        if True:
            btn_settings.clicked.connect(self._show_preferences)

        btn_console = self._build_sidebar_button(
            "Console",
            "terminal",
            checkable=False,
        )
        btn_console.setAccessibleDescription(
            "Show or hide the process output console window"
        )
        btn_console.clicked.connect(self.toggle_process_console)
        self.btn_console = btn_console
        return btn_settings, btn_console

    def _register_sidebar_buttons(
        self,
        filter_buttons: list[tuple[QToolButton, int]],
    ) -> QToolButton:
        """Register the exclusive category group and lazy Library route."""
        self.sidebar_group = QButtonGroup(self.launcher)
        for button, button_id in filter_buttons:
            self.sidebar_group.addButton(button, button_id)
        self.sidebar_group.idClicked.connect(self._on_sidebar_routed)

        btn_library = self._build_sidebar_button(
            "Library",
            "book",
            checkable=True,
        )
        btn_library.setAccessibleDescription(
            "Filter tiles to show documentation and library tools"
        )
        self.sidebar_group.addButton(btn_library, 10)
        self.btn_library_sidebar = btn_library
        self.btn_training_sidebar = None
        return btn_library

    def _populate_sidebar_layout(
        self,
        layout: QVBoxLayout,
        ordered_buttons: list[QToolButton],
        *,
        btn_console: QToolButton,
        btn_settings: QToolButton,
    ) -> None:
        """Lay out category and utility buttons with the historical spacing."""
        for index, button in enumerate(ordered_buttons):
            layout.addWidget(button)
            if index < len(ordered_buttons) - 1:
                layout.addStretch(1)
        layout.addStretch(3)
        layout.addWidget(btn_console)
        layout.addWidget(btn_settings)

    def _set_sidebar_focus_order(
        self,
        sidebar: QWidget,
        ordered_buttons: list[QToolButton],
    ) -> None:
        """Install deterministic keyboard focus traversal."""
        if not ordered_buttons:
            raise ValueError("ordered_buttons must not be empty")
        sidebar.setFocusProxy(ordered_buttons[0])
        for current, following in zip(
            ordered_buttons, ordered_buttons[1:], strict=False
        ):
            QWidget.setTabOrder(current, following)

    def _wrap_sidebar(self, sidebar: QWidget, colors: Any) -> QScrollArea:
        """Wrap the navigation widget in its resizable scroll container."""
        scroll_area = QScrollArea()
        scroll_area.setWidget(sidebar)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMinimumWidth(Styles.SIDEBAR_MIN_WIDTH)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {colors.bg_elevated};
                border-right: 1px solid {colors.border_default};
            }}
        """)
        return scroll_area

    def _on_sidebar_routed(self, button_id: int) -> None:
        """Route sidebar navigation to filter the grid layout.

        Button IDs
        ----------
        0  All (Home)
        1  Engines
        2  Biomechanics
        3  Simulation
        4  Motion Match
        5  MoCap
        6  Tools
        7  Documentation
        8  Favorites
        9  History
        10 Library
        """
        _CATEGORY_MAP: dict[int, str] = {
            0: "All",
            1: "Engines",
            2: "Biomechanics",
            3: "Simulation",
            4: "Tools",
            5: "Tools",
            6: "Tools",
            7: "Documentation",
            8: "Favorites",
            9: "History",
            10: "Documentation",
        }
        self.layout_manager.current_category_filter = _CATEGORY_MAP.get(
            button_id, "All"
        )

        self._rebuild_grid()

        if getattr(self, "workspace_tabs", None):
            self.workspace_tabs.setCurrentIndex(0)

    def _build_menu_bar_widget(self) -> QMenuBar:
        """Build a populated ``QMenuBar`` for the frameless launcher (#5624).

        Returns a standalone ``QMenuBar`` (parent ``self``) that the
        caller adds to the central widget's outer ``QVBoxLayout`` so it
        sits **below** the custom title bar.

        Postcondition: the returned widget is non-null, parented to
        ``self``, and populated with the File/View/Tools/Help menus.

        We deliberately do not call ``QMainWindow.setMenuBar`` — that
        reserves the native top strip above the central widget, which
        on a frameless window draws above the custom title bar.
        """
        menubar = QMenuBar(self.launcher)
        # Postcondition (DbC): a non-null QMenuBar is returned.
        assert menubar is not None, (
            "QMenuBar construction returned None — should be impossible"
        )

        self._setup_file_menu(menubar)
        self._setup_view_menu(menubar)
        self._setup_tools_menu(menubar)
        self._setup_help_menu(menubar)
        return menubar

    def _setup_menu_bar(self) -> None:
        """Set up the application menu bar.

        .. deprecated:: #5624
            Kept for backwards compatibility with any caller still
            reaching for the legacy hook.  Prefer
            ``_build_menu_bar_widget`` + adding the result to the
            central widget's outer layout.
        """
        menubar = self.menuBar()

        self._setup_file_menu(menubar)
        self._setup_view_menu(menubar)
        self._setup_tools_menu(menubar)
        self._setup_help_menu(menubar)
        menubar.setCornerWidget(
            _build_menu_bar_close_widget(self, self.close),
            Qt.Corner.TopRightCorner,
        )

    def _setup_file_menu(self, menubar: Any) -> None:
        if menubar is None:
            raise ValueError("menubar must be provided")
        file_menu = menubar.addMenu("&File")

        action_preferences = QAction("&Preferences...", self.launcher)
        action_preferences.setShortcut("Ctrl+,")
        action_preferences.setToolTip("Edit application preferences and theme")
        action_preferences.setStatusTip("Opens preferences")
        action_preferences.triggered.connect(self._show_preferences)
        file_menu.addAction(action_preferences)

        file_menu.addSeparator()

        action_exit = QAction("E&xit", self.launcher)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.setToolTip("Close the launcher")
        action_exit.setStatusTip("Quits the application")
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

    def _setup_view_menu(self, menubar: Any) -> None:
        if menubar is None:
            raise ValueError("menubar must be provided")
        view_menu = menubar.addMenu("&View")

        from PyQt6.QtGui import QActionGroup

        self._viewmode_action_group = QActionGroup(self.launcher)
        self._viewmode_action_group.setExclusive(True)
        self._viewmode_actions: dict[ViewMode, QAction] = {}
        for label, mode, shortcut in (
            ("Tile &Small", ViewMode.SMALL, "Ctrl+1"),
            ("Tile &Medium", ViewMode.MEDIUM, "Ctrl+2"),
            ("Tile &Large", ViewMode.LARGE, "Ctrl+3"),
            ("List &Small", ViewMode.LIST_SMALL, "Ctrl+4"),
            ("List &Large", ViewMode.LIST_LARGE, "Ctrl+5"),
        ):
            act = QAction(label, self.launcher)
            act.setCheckable(True)
            act.setShortcut(shortcut)
            act.setToolTip(f"Switch tile layout to {label.replace('&', '')} mode")
            act.setStatusTip(act.toolTip())
            act.triggered.connect(
                lambda _checked=False, m=mode: self._set_view_mode_from_menu(m)
            )
            self._viewmode_action_group.addAction(act)
            view_menu.addAction(act)
            self._viewmode_actions[mode] = act
        # Default checkmark on LIST_LARGE.
        self._viewmode_actions[ViewMode.LIST_LARGE].setChecked(True)

        view_menu.addSeparator()

        action_layout_mode = QAction("&Edit Layout Mode", self.launcher)
        action_layout_mode.setCheckable(True)
        action_layout_mode.setToolTip("Toggle drag-and-drop reordering of model tiles")
        action_layout_mode.setStatusTip("Toggles layout-edit mode")
        action_layout_mode.triggered.connect(self._toggle_layout_mode_from_menu)
        view_menu.addAction(action_layout_mode)
        self._action_layout_mode = action_layout_mode

        action_customize_tiles = QAction("&Select Visible Tiles...", self.launcher)
        action_customize_tiles.setToolTip(
            "Select which tiles are visible in the layout"
        )
        action_customize_tiles.setStatusTip("Select visible tiles")
        action_customize_tiles.triggered.connect(self.open_layout_manager)
        view_menu.addAction(action_customize_tiles)
        self.action_customize_tiles = action_customize_tiles

        view_menu.addSeparator()

        action_context_help = QAction("Context &Help Panel", self.launcher)
        action_context_help.setCheckable(True)
        action_context_help.setToolTip("Show or hide the side help panel")
        action_context_help.setStatusTip("Toggles context-help panel")
        action_context_help.triggered.connect(self._toggle_context_help)
        view_menu.addAction(action_context_help)
        self._action_context_help = action_context_help

        action_console = QAction("&Process Output Console", self.launcher)
        action_console.setCheckable(True)
        action_console.setChecked(False)
        action_console.setShortcut("Ctrl+`")
        action_console.setToolTip("Show or hide the launched-process output console")
        action_console.setStatusTip("Toggles process-output console")
        action_console.triggered.connect(lambda checked: self.toggle_process_console())
        view_menu.addAction(action_console)
        self._action_console = action_console

        view_menu.addSeparator()

        action_popout_tab = QAction("&Undock Active Tab", self.launcher)
        action_popout_tab.setShortcut("Ctrl+D")
        action_popout_tab.setToolTip("Detach the active tab into a floating window")
        action_popout_tab.setStatusTip("Undock active tab")
        action_popout_tab.triggered.connect(self._popout_active_tab)
        view_menu.addAction(action_popout_tab)
        self.action_popout_tab = action_popout_tab

        action_redock_tabs = QAction("&Redock All Tabs", self.launcher)
        action_redock_tabs.setShortcut("Ctrl+Shift+D")
        action_redock_tabs.setToolTip(
            "Redock all floating windows back to workspace tabs"
        )
        action_redock_tabs.setStatusTip("Redock all tabs")
        action_redock_tabs.triggered.connect(self._redock_all_tabs)
        view_menu.addAction(action_redock_tabs)
        self.action_redock_tabs = action_redock_tabs

        background_menu = view_menu.addMenu("&Background Tabs")
        self._background_tabs_menu = background_menu
        background_menu.aboutToShow.connect(self._refresh_background_tabs_menu)

        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("&Theme")
        self._setup_theme_menu(theme_menu)

    def _refresh_background_tabs_menu(self) -> None:
        """Populate the Background Tabs menu with restore actions (#6013).

        Lists tool tabs that were closed but kept running hidden, so the
        user can bring them back. Rebuilt each time the menu opens.
        """
        menu = getattr(self, "_background_tabs_menu", None)
        if menu is None:
            return
        menu.clear()

        tabs = getattr(self, "workspace_tabs", None)
        titles = tabs.list_background_tabs() if tabs is not None else []

        if not titles:
            empty = QAction("(no background tabs)", self.launcher)
            empty.setEnabled(False)
            menu.addAction(empty)
            return

        for title in titles:
            act = QAction(f"Restore: {title}", self.launcher)
            act.setToolTip(f"Bring the hidden '{title}' tab back into view")
            act.triggered.connect(
                lambda _checked=False, t=title: tabs.restore_background_tab(t)
            )
            menu.addAction(act)

        menu.addSeparator()
        restore_all = QAction("Restore All", self.launcher)
        restore_all.triggered.connect(tabs.restore_all_background_tabs)
        menu.addAction(restore_all)

    def _popout_active_tab(self) -> None:
        """Undock the currently active workspace tab into a floating window."""
        if not getattr(self, "workspace_tabs", None):
            return
        idx = self.workspace_tabs.currentIndex()
        if idx < 0:
            return
        tab_text = self.workspace_tabs.tabText(idx)
        if tab_text == "Home":
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.information(
                self.launcher,
                "Cannot Undock",
                "The 'Home' tab is a core view and cannot be undocked.",
            )
            return
        self.workspace_tabs.detach_tab_from_menu(idx)

    def _redock_all_tabs(self) -> None:
        """Redock all detached workspace tabs."""
        if getattr(self, "workspace_tabs", None):
            self.workspace_tabs.redock_all_tabs()

    def _setup_tools_menu(self, menubar: Any) -> None:
        if menubar is None:
            raise ValueError("menubar must be provided")
        tools_menu = menubar.addMenu("&Tools")

        action_env = QAction("&Environment Manager...", self.launcher)
        action_env.setToolTip("Inspect Python environments and engine availability")
        action_env.setStatusTip("Opens environment manager")
        action_env.triggered.connect(lambda: self._open_settings(tab=1))
        tools_menu.addAction(action_env)

        action_diag = QAction("&Diagnostics...", self.launcher)
        action_diag.setToolTip("Run a diagnostic sweep of the install")
        action_diag.setStatusTip("Opens diagnostics")
        action_diag.triggered.connect(lambda: self._open_settings(tab=2))
        tools_menu.addAction(action_diag)

    def _setup_help_menu(self, menubar: Any) -> None:
        if menubar is None:
            raise ValueError("menubar must be provided")
        help_menu = menubar.addMenu("&Help")
        self._add_primary_help_actions(help_menu)
        help_menu.addSeparator()
        self._add_topic_help_actions(help_menu)
        help_menu.addSeparator()
        self._add_support_help_actions(help_menu)

    def _add_primary_help_actions(self, help_menu: Any) -> None:
        """Add manual, contextual documentation, and project references."""
        from src.launchers.about_dialog import (
            open_motion_match_loaders_doc,
            open_user_guide,
        )

        action_manual = QAction("&User Manual", self.launcher)
        action_manual.setShortcut("F1")
        action_manual.setToolTip("Open the in-app help dialog")
        action_manual.setStatusTip("Opens user manual")
        action_manual.triggered.connect(lambda: self._show_help_dialog())
        help_menu.addAction(action_manual)

        action_context_docs = QAction("Context &Documentation", self.launcher)
        action_context_docs.setToolTip("Open context-aware documentation")
        action_context_docs.setStatusTip("Opens Context Help")
        if True:
            action_context_docs.triggered.connect(self._toggle_context_help)
        help_menu.addAction(action_context_docs)

        action_user_guide = QAction("User &Guide (online)", self.launcher)
        action_user_guide.setToolTip(
            "Open the bundled user guide in the system browser"
        )
        action_user_guide.setStatusTip("Opens user guide")
        action_user_guide.triggered.connect(lambda: open_user_guide())
        help_menu.addAction(action_user_guide)

        action_loaders = QAction("&Motion-Match Loaders", self.launcher)
        action_loaders.setToolTip("Reference for loading motion-target files")
        action_loaders.setStatusTip("Opens motion-match loader reference")
        action_loaders.triggered.connect(lambda: open_motion_match_loaders_doc())
        help_menu.addAction(action_loaders)

        action_project_map = QAction("&Project Map", self.launcher)
        action_project_map.setToolTip("Open the project-map document")
        action_project_map.setStatusTip("Opens project map")
        action_project_map.triggered.connect(self._open_project_map)
        help_menu.addAction(action_project_map)

    def _add_topic_help_actions(self, help_menu: Any) -> None:
        """Add context-topic actions with stable lambda capture."""
        topic_tips: dict[str, tuple[str, str]] = {
            "engine_selection": (
                "How to choose between the bundled physics engines",
                "Opens engine-selection help",
            ),
            "simulation_controls": (
                "Reference for simulation playback and stepping controls",
                "Opens simulation-controls help",
            ),
            "motion_capture": (
                "Loading and aligning motion-capture targets",
                "Opens motion-capture help",
            ),
            "visualization": (
                "Camera, traces, and scene-element controls",
                "Opens visualization help",
            ),
            "analysis_tools": (
                "Built-in analysis and post-processing tools",
                "Opens analysis-tools help",
            ),
        }
        for label, topic in (
            ("Engine &Selection Guide", "engine_selection"),
            ("Simulation &Controls", "simulation_controls"),
            ("Motion &Capture", "motion_capture"),
            ("&Visualization", "visualization"),
            ("&Analysis Tools", "analysis_tools"),
        ):
            action = QAction(label, self.launcher)
            tip, status = topic_tips[topic]
            action.setToolTip(tip)
            action.setStatusTip(status)
            action.triggered.connect(
                lambda checked, topic_name=topic: self._show_help_dialog(topic_name)
            )
            help_menu.addAction(action)

    def _add_support_help_actions(self, help_menu: Any) -> None:
        """Add keyboard-shortcut, issue-reporting, and About actions."""
        from src.launchers.about_dialog import open_issues_page, show_about_dialog
        from src.launchers.help_menu import show_keyboard_shortcuts_modal

        action_shortcuts = QAction("&Keyboard Shortcuts...", self.launcher)
        action_shortcuts.setShortcut("Ctrl+?")
        action_shortcuts.setToolTip("Show every registered keyboard shortcut")
        action_shortcuts.setStatusTip("Opens keyboard-shortcuts table")

        def _open_shortcuts() -> None:
            try:
                show_keyboard_shortcuts_modal(self.launcher)
            except Exception:  # noqa: BLE001
                self._show_shortcuts_overlay()

        action_shortcuts.triggered.connect(_open_shortcuts)
        help_menu.addAction(action_shortcuts)

        action_report_bug = QAction("&Report a Bug...", self.launcher)
        action_report_bug.setToolTip("Open the public issue tracker in your browser")
        action_report_bug.setStatusTip("Opens issue tracker")
        action_report_bug.triggered.connect(lambda: open_issues_page())
        help_menu.addAction(action_report_bug)
        help_menu.addSeparator()

        action_about = QAction("&About UpstreamDrift", self.launcher)
        action_about.setToolTip("Show version and runtime information")
        action_about.setStatusTip("Opens About dialog")

        def _open_about() -> None:
            try:
                show_about_dialog(self.launcher)
            except Exception:  # noqa: BLE001
                self._show_about_dialog()

        action_about.triggered.connect(_open_about)
        help_menu.addAction(action_about)
