"""Launcher top-bar, runtime status, view-mode, and zoom behavior (#8490)."""

# mypy: disable-error-code="attr-defined,call-overload,arg-type,assignment"

from __future__ import annotations

import contextlib
from typing import Any

from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QEnterEvent,
    QKeySequence,
    QMouseEvent,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QWidget,
)

from src.launchers.launcher_constants import TILE_SCALE_MAX, TILE_SCALE_MIN, ViewMode
from src.shared.python.theme.typography import Weights, get_display_font
from src.shared.python.ui.auto_complete import AutoCompleteLineEdit
from src.shared.python.ui.completion_vocab import build_vocabulary


def _resolve_theme_color(c_obj: Any, ns_key: str, dict_key: str, default: str) -> str:
    """Helper to resolve colors from namespace/dict theme tokens."""
    if isinstance(c_obj, dict):
        return c_obj.get(dict_key, default)
    return getattr(c_obj, ns_key, getattr(c_obj, dict_key, default))


class ClickableLabel(QLabel):
    """QLabel subclass that emits a clicked signal on mouse click."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)

    def enterEvent(self, event: QEnterEvent) -> None:
        if self.cursor().shape() == Qt.CursorShape.PointingHandCursor:
            font = self.font()
            font.setUnderline(True)
            self.setFont(font)
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        font = self.font()
        font.setUnderline(False)
        self.setFont(font)
        super().leaveEvent(event)


class RuntimeButton(QToolButton):
    """QToolButton subclass designed to mimic model card tiles.

    Supports custom drop shadow, theme-matching styling, and deferred hover-hide behavior
    for its associated help button.
    """

    def __init__(
        self, parent: QWidget | None = None, help_button: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.help_button = help_button
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("RuntimeButton")
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        # Reduced/extremely subtle shadow
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(2)
        self.shadow.setOffset(0, 1)
        self.shadow.setColor(QColor(0, 0, 0, 15))
        self.setGraphicsEffect(self.shadow)

        # Deferred hide timer to allow moving mouse to the help button
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(300)  # 300ms delay
        self._hide_timer.timeout.connect(self._do_hide)

    def enterEvent(self, event: QEnterEvent) -> None:
        self._hide_timer.stop()
        if self.help_button:
            self.help_button.show()
            self.help_button.raise_()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self._hide_timer.start()
        super().leaveEvent(event)

    def cancel_pending_hide(self) -> None:
        """Keep the runtime help affordance visible while it is hovered."""
        self._hide_timer.stop()

    def schedule_hide(self) -> None:
        """Hide the runtime help affordance after the configured delay."""
        self._hide_timer.start()

    def _do_hide(self) -> None:
        if self.help_button and not self.help_button.underMouse():
            self.help_button.hide()


class HelpButtonHoverFilter(QObject):
    """Event filter to prevent hiding the help button when hovered directly."""

    def __init__(self, runtime_button: RuntimeButton) -> None:
        super().__init__()
        self.runtime_button = runtime_button

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Enter:
            self.runtime_button.cancel_pending_hide()
        elif event.type() == QEvent.Type.Leave:
            self.runtime_button.schedule_hide()
        return super().eventFilter(obj, event)


def _build_zoom_accessible_description(
    minimum_scale: float = TILE_SCALE_MIN,
    maximum_scale: float = TILE_SCALE_MAX,
) -> str:
    """Describe the zoom slider using the configured tile-scale bounds."""
    minimum_pct = int(round(minimum_scale * 100))
    maximum_pct = int(round(maximum_scale * 100))
    return (
        f"Adjust tile size from {minimum_pct}% to {maximum_pct}%. "
        "Use arrow keys or drag to adjust."
    )


class LauncherTopBarUIMixin:
    """Build and coordinate the launcher's top-bar controls."""

    def _setup_top_bar_status_and_search(self, top_bar: QHBoxLayout) -> None:
        """Add status indicator, execution mode label, and search bar."""
        if top_bar is None:
            raise ValueError("top_bar must be provided")
        self._setup_status_indicator(top_bar)
        self._setup_runtime_indicator(top_bar)
        top_bar.addStretch()
        self._setup_search_controls(top_bar)
        self._ensure_launch_button()

    def _setup_status_indicator(self, top_bar: QHBoxLayout) -> None:
        """Create the clickable launcher status indicator."""
        self.lbl_status = ClickableLabel("Checking Docker...")
        self.lbl_status.setProperty("status", "inactive-bold")
        style = self.lbl_status.style()
        if style:
            style.polish(self.lbl_status)
        self.lbl_status.clicked.connect(self._on_status_clicked)
        top_bar.addWidget(self.lbl_status)

    def _setup_runtime_indicator(self, top_bar: QHBoxLayout) -> None:
        """Create the engine-runtime control and its hover help."""
        from src.launchers.runtime_mode_help import make_runtime_mode_help_button

        self.btn_runtime_help = make_runtime_mode_help_button(self.launcher)
        self.btn_runtime_help.hide()

        try:
            import src.shared.python.theme as theme

            colors = theme.get_current_colors()  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            from src.shared.python.theme import DARK_THEME as colors

        background = _resolve_theme_color(
            colors, "surface_hover", "group_bg", "#2d2d2d"
        )
        border = _resolve_theme_color(colors, "border_light", "border", "#444444")
        hover_background = _resolve_theme_color(
            colors, "surface_active", "input_bg", "#3a3a3a"
        )
        hover_border = _resolve_theme_color(colors, "border_strong", "focus", "#666666")
        text_color = _resolve_theme_color(colors, "text_primary", "text", "#ffffff")

        self.lbl_execution_mode = RuntimeButton(help_button=self.btn_runtime_help)
        self.lbl_execution_mode.setText("Runtime: Windows")
        self.lbl_execution_mode.setProperty("exec_mode", "warning")
        style = self.lbl_execution_mode.style()
        if style:
            style.polish(self.lbl_execution_mode)
        self.lbl_execution_mode.setToolTip(
            "Click to change the engine runtime (Windows, Docker, or "
            "WSL2 Ubuntu) in Settings → Configuration. Hover to explain."
        )
        self.lbl_execution_mode.setStyleSheet(f"""
            QToolButton#RuntimeButton {{
                background-color: {background};
                border: 1px solid {border};
                border-radius: 8px;
                padding: 4px 12px;
                color: {text_color};
                font-weight: bold;
                margin-left: 10px;
            }}
            QToolButton#RuntimeButton:hover {{
                background-color: {hover_background};
                border: 1px solid {hover_border};
            }}
        """)
        self.lbl_execution_mode.clicked.connect(
            lambda: self.launcher._open_settings(tab=1)
        )
        top_bar.addWidget(self.lbl_execution_mode)

        self._help_button_filter = HelpButtonHoverFilter(self.lbl_execution_mode)
        self.btn_runtime_help.installEventFilter(self._help_button_filter)
        top_bar.addWidget(self.btn_runtime_help)

    def _setup_search_controls(self, top_bar: QHBoxLayout) -> None:
        """Create model search and clear-filter controls."""
        self.search_input = AutoCompleteLineEdit(words=build_vocabulary())
        self.search_input.setPlaceholderText("Search models... (Esc to clear)")
        try:
            from src.shared.python.theme.responsive import (
                TextWidthSpec,
                set_text_minimum_width,
            )

            set_text_minimum_width(
                self.search_input,
                TextWidthSpec(minimum_px=250),
            )
        except ImportError:
            self.search_input.setFixedWidth(250)
        self.search_input.setToolTip("Filter models by name or description (Ctrl+F)")
        self.search_input.setAccessibleName("Search models")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self.update_search_filter)
        top_bar.addWidget(self.search_input)

        self.btn_clear_filters = QPushButton("Clear Filters")
        self.btn_clear_filters.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_filters.setToolTip(
            "Reset all category and search filters to show all tiles"
        )
        self.btn_clear_filters.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 4px;
                padding: 4px 10px;
                color: #e0e0e0;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.12);
                border-color: rgba(255, 255, 255, 0.25);
                color: #ffffff;
            }
        """)
        self.btn_clear_filters.clicked.connect(self._clear_all_filters)
        self.btn_clear_filters.hide()
        top_bar.addWidget(self.btn_clear_filters)

    def _on_status_clicked(self) -> None:
        """Handle clicking the status label. If in error/dependency error, opens environment manager.
        If there are running processes, routes to the Processes tab in Settings.
        """
        status_text = self.lbl_status.text()
        if "Dependency Error" in status_text or "Error" in status_text:
            self.launcher.open_environment_manager()
            return

        # Check active processes
        running = []
        with self.process_manager._process_lock:
            for name, proc in list(self.running_processes.items()):
                if proc.poll() is None:
                    running.append(name)

        if running:
            self._open_settings(tab=8)

    def _ensure_launch_button(self) -> None:
        """Create ``self.btn_launch`` if it doesn't exist yet (idempotent).

        Used by both ``_setup_top_bar_status_and_search`` (current home) and
        ``_setup_bottom_bar`` (legacy callers / tests). Either path produces
        the same button instance.
        """
        if getattr(self, "btn_launch", None) is not None:
            return
        btn = QPushButton("Select a Model")
        btn.setEnabled(False)
        # Top-bar height: align with search input rather than the old 50px
        # tile-overlapping bottom bar. Width caps the button so a long model
        # name (e.g. "Launch golf_swing_pendulum >") doesn't push the rest of
        # the top bar offscreen.
        btn.setFixedHeight(32)
        btn.setMinimumWidth(180)
        btn.setMaximumWidth(280)
        btn.setFont(get_display_font(size=10, weight=Weights.BOLD))
        btn.setProperty("class", "launch-ready")
        _style = btn.style()
        if _style:
            _style.polish(btn)
        btn.clicked.connect(self.launch_simulation)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.hide()  # Hidden per user request; models are launched via tiles.
        self.btn_launch = btn

    def _setup_top_bar_config_checkboxes(self, top_bar: QHBoxLayout) -> None:
        """Create config checkboxes and layout controls, adding them to top bar."""
        from PyQt6.QtCore import QSettings

        settings = QSettings("UpstreamDrift", "Launcher")

        self.chk_live = QCheckBox("Live Viz")
        self.chk_live.setChecked(settings.value("chk_live", True, type=bool))

        self.chk_gpu = QCheckBox("GPU")
        self.chk_gpu.setChecked(settings.value("chk_gpu", False, type=bool))

        self.chk_docker = QCheckBox("Docker")
        docker_default = getattr(self, "docker_available", False)
        self.chk_docker.setChecked(
            settings.value("chk_docker", docker_default, type=bool)
        )
        self.chk_docker.stateChanged.connect(self._on_docker_mode_changed)

        self.chk_wsl = QCheckBox("WSL")
        self.chk_wsl.setChecked(settings.value("chk_wsl", False, type=bool))
        self.chk_wsl.stateChanged.connect(self._on_wsl_mode_changed)

        self.chk_windows = QCheckBox("Windows")
        self.chk_windows.setChecked(
            not self.chk_docker.isChecked() and not self.chk_wsl.isChecked()
        )
        self.chk_windows.stateChanged.connect(self._on_windows_mode_changed)

    def _setup_top_bar_action_buttons(self, top_bar: QHBoxLayout) -> None:
        """Add Help, Settings, and AI Assistant buttons to top bar."""

    def _register_top_bar_tooltips(self) -> None:
        """Register enhanced tooltips for configuration checkboxes."""
        from src.launchers.launcher_constants import HELP_SYSTEM_AVAILABLE

        if not HELP_SYSTEM_AVAILABLE:
            return

        from src.shared.python.gui_pkg.help_system import TooltipManager

        TooltipManager.register_tooltip(
            self.chk_live,
            "Live Visualization",
            "Enable real-time 3D visualization during simulation.",
            "visualization",
        )
        TooltipManager.register_tooltip(
            self.chk_gpu,
            "GPU Acceleration",
            "Use GPU for physics computation when available.",
            "engine_selection",
        )
        TooltipManager.register_tooltip(
            self.chk_docker,
            "Docker container runtime",
            "Run physics engines inside the upstream-drift:engine Linux "
            "container. Full Drake/Pinocchio support; requires Docker "
            "installed and the image built (Settings → Docker Image).",
            "engine_selection",
        )
        TooltipManager.register_tooltip(
            self.chk_wsl,
            "WSL2 Ubuntu runtime",
            "Run physics engines in your WSL2 Ubuntu user environment. "
            "Same Linux wheels as Docker mode but no container layer — "
            "faster file I/O and easier interactive debugging.",
            "engine_selection",
        )
        TooltipManager.register_tooltip(
            self.chk_windows,
            "Windows Native runtime",
            "Run physics engines natively on the local Windows host system.",
            "engine_selection",
        )

    def _slider_to_scale(self, value: int) -> float:
        """Map a slider integer ``value`` to a tile_scale float."""
        v = max(0, min(self._ZOOM_SLIDER_STEPS, int(value)))
        frac = v / float(self._ZOOM_SLIDER_STEPS)
        return TILE_SCALE_MIN + (TILE_SCALE_MAX - TILE_SCALE_MIN) * frac

    def _scale_to_slider(self, scale: float) -> int:
        """Map a tile_scale float back to slider integer steps."""
        s = max(TILE_SCALE_MIN, min(TILE_SCALE_MAX, float(scale)))
        frac = (s - TILE_SCALE_MIN) / (TILE_SCALE_MAX - TILE_SCALE_MIN)
        return int(round(frac * self._ZOOM_SLIDER_STEPS))

    def _setup_view_mode_and_zoom(self, top_bar: QHBoxLayout) -> None:
        """Create the hidden view-mode and zoom compatibility controls."""
        if top_bar is None:
            raise ValueError("top_bar must be provided")
        self._top_viewmode_actions: dict[Any, QAction] = {}
        self._setup_view_mode_combo()
        self._setup_zoom_slider()
        self._setup_zoom_shortcuts()

    def _setup_view_mode_combo(self) -> None:
        """Create the hidden view-mode selector used by menu synchronization."""
        self.view_mode_combo = QComboBox(self.launcher)
        self.view_mode_combo.addItem("Tile Large", ViewMode.LARGE)
        self.view_mode_combo.addItem("Tile Medium", ViewMode.MEDIUM)
        self.view_mode_combo.addItem("Tile Small", ViewMode.SMALL)
        self.view_mode_combo.addItem("List Large", ViewMode.LIST_LARGE)
        self.view_mode_combo.addItem("List Small", ViewMode.LIST_SMALL)
        self.view_mode_combo.setCurrentIndex(3)
        self.view_mode_combo.setToolTip("Choose how the model tiles are arranged")
        self.view_mode_combo.setAccessibleName("View mode")
        self.view_mode_combo.currentIndexChanged.connect(self._on_view_mode_changed)
        self.view_mode_combo.setStyleSheet("""
            QComboBox {
                background: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 4px 8px;
                color: #cccccc;
                font-size: 11px;
                min-width: 100px;
            }
            QComboBox:hover {
                background: #2a2a2a;
                border-color: #555555;
                color: #ffffff;
            }
            QComboBox QAbstractItemView {
                background: #1e1e1e;
                border: 1px solid #3a3a3a;
                color: #cccccc;
                selection-background-color: #2a2a2a;
                selection-color: #ffffff;
            }
        """)
        self.view_mode_combo.hide()

    def _setup_zoom_slider(self) -> None:
        """Create hidden zoom controls and restore the current tile scale."""
        from PyQt6.QtWidgets import QSizePolicy

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal, self.launcher)
        self.zoom_slider.setRange(0, self._ZOOM_SLIDER_STEPS)
        self.zoom_slider.setMinimumWidth(140)
        self.zoom_slider.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            self.zoom_slider.sizePolicy().verticalPolicy(),
        )
        self.zoom_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #3a3a3a;
                height: 4px;
                background: #1a1a1a;
                margin: 0px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #888888;
                border: 1px solid #555555;
                width: 10px;
                height: 10px;
                margin: -3px 0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal:hover {
                background: #007acc;
                border-color: #0098ff;
            }
        """)
        self.zoom_slider.setToolTip("Adjust the size of the model tiles")
        self.zoom_slider.setAccessibleName("Tile zoom")
        self.zoom_slider.setAccessibleDescription(_build_zoom_accessible_description())
        self.zoom_slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        from src.launchers.launcher_constants import TILE_SCALE_DEFAULT

        initial_scale = TILE_SCALE_DEFAULT
        layout_manager = getattr(self, "layout_manager", None)
        if layout_manager is not None and hasattr(layout_manager, "tile_scale"):
            initial_scale = float(layout_manager.tile_scale)
        self.zoom_slider.setValue(self._scale_to_slider(initial_scale))
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        self.zoom_slider.hide()

        self.lbl_zoom_pct = QLabel(
            f"{int(round(initial_scale * 100))}%",
            self.launcher,
        )
        self.lbl_zoom_pct.setToolTip("Current tile size as a percentage of base")
        self.lbl_zoom_pct.setStyleSheet(
            "font-size: 10px; color: #888888; font-family: monospace;"
        )
        self.lbl_zoom_pct.hide()

    def _setup_zoom_shortcuts(self) -> None:
        """Register the historical keyboard zoom bindings."""
        shortcut_in = QShortcut(QKeySequence("Ctrl+="), self.launcher)
        shortcut_in.activated.connect(lambda: self._nudge_zoom(+5))
        shortcut_in_alt = QShortcut(QKeySequence("Ctrl++"), self.launcher)
        shortcut_in_alt.activated.connect(lambda: self._nudge_zoom(+5))
        shortcut_out = QShortcut(QKeySequence("Ctrl+-"), self.launcher)
        shortcut_out.activated.connect(lambda: self._nudge_zoom(-5))

    def _nudge_zoom(self, delta_steps: int) -> None:
        """Adjust the zoom slider by ``delta_steps`` integer ticks."""
        slider = self.zoom_slider
        if slider is None:
            return
        slider.setValue(slider.value() + delta_steps)

    def _on_view_mode_changed(self, index: int) -> None:
        """Apply the selected view mode to the layout manager + grid."""
        combo = self.view_mode_combo
        if combo is None:
            return
        mode = combo.itemData(index)
        is_vm = isinstance(mode, ViewMode) or (
            hasattr(mode, "__class__") and mode.__class__.__name__ == "ViewMode"
        )
        if not is_vm:
            return
        if not isinstance(mode, ViewMode):
            try:
                mode = ViewMode(int(mode))
            except (ValueError, TypeError):
                with contextlib.suppress(AttributeError, KeyError):
                    mode = ViewMode[mode.name]
        self._apply_view_mode(mode, sync_combo=False)

    def _set_view_mode_from_menu(self, mode: ViewMode) -> None:
        """Drive the view-mode change from the View menu's submenu."""
        self._apply_view_mode(mode, sync_combo=True)

    def _apply_view_mode(self, mode: ViewMode, *, sync_combo: bool) -> None:
        """Single source of truth for changing tile layout mode.

        Keeps the menubar action group, the top-bar dropdown, the zoom
        slider, and the grid in sync regardless of which surface
        triggered the change.
        """
        lm = self.layout_manager
        if lm is None:
            return
        lm.set_view_mode(mode)
        # Sync menu action checkmarks regardless.
        actions = self._viewmode_actions
        if actions and mode in actions and not actions[mode].isChecked():
            actions[mode].setChecked(True)
        # Sync top-bar dropdown menu action checkmarks regardless.
        top_actions = self._top_viewmode_actions
        if top_actions and mode in top_actions and not top_actions[mode].isChecked():
            top_actions[mode].setChecked(True)
        # Update zoom slider/label to reflect the mode's default scale.
        if True:
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(self._scale_to_slider(lm.tile_scale))
            self.zoom_slider.blockSignals(False)
        if True:
            self.lbl_zoom_pct.setText(f"{int(round(lm.tile_scale * 100))}%")
        if True:
            lm.rebuild_grid(self.grid_layout)
        if True:
            self._save_layout()

    def _on_zoom_slider_changed(self, value: int) -> None:
        """Live-resize all model cards to match the new slider position."""
        lm = self.layout_manager
        scale = self._slider_to_scale(value)
        if True:
            self.lbl_zoom_pct.setText(f"{int(round(scale * 100))}%")
        if lm is None:
            return
        lm.set_tile_scale(scale)
        if True:
            self._rebuild_grid()
        if True:
            self._save_layout()

    def _setup_top_bar(self) -> QHBoxLayout:
        """Set up the top tool bar."""
        top_bar = QHBoxLayout()

        # Modern toggles for the sidebars (left nav and right sidekick)
        self.btn_toggle_left_sidebar = QToolButton(self.launcher)
        try:
            from src.shared.python.theme.icon_utils import IconColorizer

            self.btn_toggle_left_sidebar.setIcon(
                IconColorizer.get_icon("menu", "#cccccc")
            )
        except ImportError:
            self.btn_toggle_left_sidebar.setText("☰")
        self.btn_toggle_left_sidebar.setToolTip("Toggle Navigation Sidebar")
        self.btn_toggle_left_sidebar.setCheckable(True)
        self.btn_toggle_left_sidebar.setChecked(True)
        self.btn_toggle_left_sidebar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_left_sidebar.setStyleSheet(
            "QToolButton { background: transparent; padding: 4px 8px; border-radius: 4px; } QToolButton:hover { background: #2a2a2a; }"
        )
        self.btn_toggle_left_sidebar.clicked.connect(self._toggle_left_sidebar)
        top_bar.addWidget(self.btn_toggle_left_sidebar)

        self._setup_top_bar_status_and_search(top_bar)
        self._setup_view_mode_and_zoom(top_bar)
        self._setup_top_bar_config_checkboxes(top_bar)
        self._setup_top_bar_action_buttons(top_bar)

        self.btn_toggle_right_sidebar = QToolButton(self.launcher)
        try:
            from src.shared.python.theme.icon_utils import IconColorizer

            self.btn_toggle_right_sidebar.setIcon(
                IconColorizer.get_icon("chat", "#cccccc")
            )
        except ImportError:
            self.btn_toggle_right_sidebar.setText("💬")
        self.btn_toggle_right_sidebar.setToolTip("Toggle Sidekick Chat")
        self.btn_toggle_right_sidebar.setCheckable(True)
        self.btn_toggle_right_sidebar.setChecked(True)
        self.btn_toggle_right_sidebar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_right_sidebar.setStyleSheet(
            "QToolButton { background: transparent; padding: 4px 8px; border-radius: 4px; } QToolButton:hover { background: #2a2a2a; }"
        )
        self.btn_toggle_right_sidebar.clicked.connect(self._toggle_sidekick)
        top_bar.addWidget(self.btn_toggle_right_sidebar)

        # Context Help Dock
        self._setup_context_help()

        # Register enhanced tooltips
        self._register_top_bar_tooltips()

        return top_bar
