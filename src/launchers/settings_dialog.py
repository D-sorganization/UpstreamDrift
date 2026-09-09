"""Settings dialog for the UpstreamDrift Launcher.

Provides a tabbed dialog with Layout, Configuration, and Diagnostics tabs.
"""

# mypy: disable-error-code="attr-defined,assignment,union-attr,arg-type"

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.launchers.settings_close_contract import SettingsCloseContract
from src.launchers.docker_manager import DockerBuildThread
from src.launchers.docker_profile_info import load_docker_profiles
from src.launchers.launcher_constants import DOCKER_STAGES
from src.launchers._settings_auxiliary_tabs import SettingsAuxiliaryTabsMixin
from src.launchers.settings_runtime import (
    RuntimeDependencyCheckFailure as RuntimeDependencyCheckFailure,
    RuntimeDependencyCheckWorker,
    RuntimeDependencyReport,
    WslScriptDialog,
    check_docker_dependencies_report as _check_docker_dependencies_report,
    check_wsl_dependencies_report as _check_wsl_dependencies_report,
    compare_version_strings as _compare_version_strings,
)
from src.shared.python.docker_config import DOCKER_IMAGE_ENGINE as DOCKER_IMAGE_NAME
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

from .startup import REPOS_ROOT

logger = get_logger(__name__)

TAB_LAYOUT = 0
TAB_CONFIG = 1
TAB_DIAGNOSTICS = 2
TAB_MCP_SERVERS = 3
TAB_APPEARANCE = 4
TAB_STARTUP = 5
TAB_NOTIFICATIONS = 6
TAB_PERFORMANCE = 7
TAB_PROCESSES = 8


def validate_tab_index(tab_index: int) -> int:
    """Validate SettingsDialog startup tab index."""
    valid_indexes = {
        TAB_LAYOUT,
        TAB_CONFIG,
        TAB_DIAGNOSTICS,
        TAB_MCP_SERVERS,
        TAB_APPEARANCE,
        TAB_STARTUP,
        TAB_NOTIFICATIONS,
        TAB_PERFORMANCE,
        TAB_PROCESSES,
    }
    if tab_index not in valid_indexes:
        raise ValueError(
            f"Invalid tab index {tab_index}; expected one of {sorted(valid_indexes)}"
        )
    return tab_index


class SettingsWidget(SettingsCloseContract, SettingsAuxiliaryTabsMixin, QWidget):
    """Settings widget with Layout, Configuration, Diagnostics, MCP Servers, and Preferences tabs.

    Tab order:
        0 - Layout: tile arrangement, lock, reset
        1 - Configuration: execution env, simulation opts, Docker rebuild
        2 - Diagnostics: system checks, error logs, terminal output
        3 - MCP Servers: manage Model Context Protocol server connections
        4 - Appearance
        5 - Startup
        6 - Notifications
        7 - Performance
    """

    reset_layout_requested = pyqtSignal()

    # Tab index constants for external callers
    TAB_LAYOUT = TAB_LAYOUT
    TAB_CONFIG = TAB_CONFIG
    TAB_DIAGNOSTICS = TAB_DIAGNOSTICS
    TAB_MCP_SERVERS = TAB_MCP_SERVERS
    TAB_APPEARANCE = TAB_APPEARANCE
    TAB_STARTUP = TAB_STARTUP
    TAB_NOTIFICATIONS = TAB_NOTIFICATIONS
    TAB_PERFORMANCE = TAB_PERFORMANCE
    TAB_PROCESSES = TAB_PROCESSES

    #: Tabs that commit only via Apply Preferences (#8896).
    preference_tab_indexes = frozenset(
        {TAB_APPEARANCE, TAB_STARTUP, TAB_NOTIFICATIONS, TAB_PERFORMANCE}
    )

    def __init__(
        self,
        parent: QWidget | None = None,
        diagnostics_data: dict[str, Any] | None = None,
        initial_tab: int = 0,
        launcher: Any | None = None,
    ) -> None:
        if initial_tab is None:
            raise ValueError("initial_tab must be provided")
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(850, 650)
        self._diagnostics_data = diagnostics_data
        self._launcher = launcher or parent
        self._diagnostics_loaded = False
        self._docker_profile_infos = load_docker_profiles()
        self._dep_check_workers: dict[str, RuntimeDependencyCheckWorker] = {}
        self._setup_ui()
        self.tabs.setCurrentIndex(validate_tab_index(initial_tab))
        # Connect tab change signal for lazy diagnostics loading
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.currentChanged.connect(self.sync_commit_model_caption)
        self.sync_commit_model_caption(self.tabs.currentIndex())

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        from src.shared.python.gui_pkg.draggable_tabs import DraggableTabWidget

        self.tabs = DraggableTabWidget(
            core_tabs={
                "Layout",
                "Configuration",
                "Diagnostics",
                "MCP Servers",
                "Appearance",
                "Startup",
                "Notifications",
                "Performance",
                "Processes",
            }
        )
        self.tabs.setTabsClosable(False)
        layout.addWidget(self.tabs)
        self.install_commit_model_caption(layout)

        self.tabs.addTab(self._create_layout_tab(), "Layout")
        self.tabs.addTab(self._create_configuration_tab(), "Configuration")
        self.tabs.addTab(self._create_diagnostics_tab(), "Diagnostics")
        self.tabs.addTab(self._create_mcp_servers_tab(), "MCP Servers")

        # Load preferences dialog tabs
        try:
            from src.shared.python.ui.preferences_dialog import PreferencesDialog

            prefs_parent = self._launcher or self
            self._prefs_dialog = PreferencesDialog(prefs_parent)
            self.tabs.addTab(self._prefs_dialog._create_appearance_tab(), "Appearance")
            self.tabs.addTab(self._prefs_dialog._create_startup_tab(), "Startup")
            self.tabs.addTab(
                self._prefs_dialog._create_notifications_tab(), "Notifications"
            )
            self.tabs.addTab(
                self._prefs_dialog._create_performance_tab(), "Performance"
            )
            self.tabs.addTab(self._create_processes_tab(), "Processes")

            self.install_apply_button(layout)
        except ImportError:
            self.tabs.addTab(self._create_processes_tab(), "Processes")

    # ── Layout tab ──────────────────────────────────────────────────

    def _create_layout_tab(self) -> QWidget:
        """Layout tab: tile lock, edit tiles, reset to defaults."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        tab_layout.addWidget(self._build_tile_layout_group())
        tab_layout.addWidget(self._build_view_mode_group())
        tab_layout.addWidget(self._build_tile_zoom_group())
        self._sync_layout_tab_with_launcher()

        tab_layout.addStretch()
        return tab

    def _build_tile_layout_group(self) -> QGroupBox:
        """Build the "Tile Layout" group: lock toggle, edit tiles, reset."""
        group = QGroupBox("Tile Layout")
        inner = QVBoxLayout(group)

        self._btn_layout_lock = QPushButton("Layout: Locked")
        self._btn_layout_lock.setCheckable(True)
        self._btn_layout_lock.setChecked(False)
        self._btn_layout_lock.setStyleSheet(Styles.BTN_LAYOUT_TOGGLE)
        self._btn_layout_lock.setToolTip(
            "Layout is locked. Click to unlock layout for editing."
        )
        self._btn_layout_lock.toggled.connect(self._on_layout_lock_toggled)
        inner.addWidget(self._btn_layout_lock)

        self._btn_edit_tiles = QPushButton("Edit Tiles (show/hide)")
        self._btn_edit_tiles.setEnabled(False)
        self._btn_edit_tiles.setToolTip(
            "Unlock the layout above to edit which tiles are visible"
        )
        inner.addWidget(self._btn_edit_tiles)

        inner.addSpacing(12)

        btn_reset = QPushButton("Reset Layout to Defaults")
        btn_reset.setToolTip("Restore all tiles and default arrangement")
        btn_reset.clicked.connect(self._on_reset_layout)
        inner.addWidget(btn_reset)

        return group

    def _build_view_mode_group(self) -> QGroupBox:
        """Build the "View Mode" group with the tile/list view combo box."""
        view_mode_group = QGroupBox("View Mode")
        view_mode_inner = QVBoxLayout(view_mode_group)

        self.combo_view_mode = QComboBox()
        # Ensure enums are imported correctly inside the method scope
        try:
            from src.launchers.launcher_constants import ViewMode

            self.combo_view_mode.addItem("Tile Small", ViewMode.SMALL)
            self.combo_view_mode.addItem("Tile Medium", ViewMode.MEDIUM)
            self.combo_view_mode.addItem("Tile Large", ViewMode.LARGE)
            self.combo_view_mode.addItem("List Small", ViewMode.LIST_SMALL)
            self.combo_view_mode.addItem("List Large", ViewMode.LIST_LARGE)
        except ImportError:
            pass

        view_mode_inner.addWidget(self.combo_view_mode)
        return view_mode_group

    def _build_tile_zoom_group(self) -> QGroupBox:
        """Build the "Tile Zoom" group with the tile-size slider."""
        zoom_group = QGroupBox("Tile Zoom")
        zoom_inner = QHBoxLayout(zoom_group)

        from PyQt6.QtWidgets import QSlider

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(0, 100)
        self.zoom_slider.setToolTip("Adjust the size of the model tiles")

        self.lbl_zoom_pct = QLabel("100%")
        self.lbl_zoom_pct.setFixedWidth(35)
        self.lbl_zoom_pct.setStyleSheet("color: #888888; font-family: monospace;")

        zoom_inner.addWidget(QLabel("Smaller"))
        zoom_inner.addWidget(self.zoom_slider)
        zoom_inner.addWidget(QLabel("Larger"))
        zoom_inner.addSpacing(10)
        zoom_inner.addWidget(self.lbl_zoom_pct)
        return zoom_group

    def _sync_layout_tab_with_launcher(self) -> None:
        """Wire the layout tab's widgets to the parent launcher's live state."""
        launcher = self._launcher
        # ``btn_modify_layout`` no longer exists (issue #8023); the checkable
        # ``View > Edit Layout Mode`` QAction owns the layout-edit state.
        layout_action = getattr(launcher, "_action_layout_mode", None)
        if launcher and layout_action is not None:
            is_unlocked = layout_action.isChecked()
            self._btn_layout_lock.setChecked(is_unlocked)
            self._on_layout_lock_toggled(is_unlocked)
            self._btn_layout_lock.toggled.connect(layout_action.setChecked)
            self._btn_layout_lock.toggled.connect(
                launcher._toggle_layout_mode_from_menu
            )
            layout_action.toggled.connect(self._sync_layout_lock_state)
            self._btn_edit_tiles.clicked.connect(launcher.open_layout_manager)

            # Sync view mode
            if hasattr(launcher, "layout_manager"):
                current_mode = launcher.layout_manager.view_mode
                index = self.combo_view_mode.findData(current_mode)
                if index >= 0:
                    self.combo_view_mode.setCurrentIndex(index)

                self.combo_view_mode.currentIndexChanged.connect(
                    lambda idx: launcher._set_view_mode_from_menu(
                        self.combo_view_mode.itemData(idx)
                    )
                )

            # Sync zoom
            if hasattr(launcher, "layout_manager"):
                scale = launcher.layout_manager.tile_scale
                val = (
                    launcher._scale_to_slider(scale)
                    if hasattr(launcher, "_scale_to_slider")
                    else 50
                )
                self.zoom_slider.setValue(val)
                self.lbl_zoom_pct.setText(f"{int(round(scale * 100))}%")

                def on_zoom(v):
                    if hasattr(launcher, "_slider_to_scale"):
                        self.lbl_zoom_pct.setText(
                            f"{int(round(launcher._slider_to_scale(v) * 100))}%"
                        )
                    if hasattr(launcher, "_on_zoom_slider_changed"):
                        launcher._on_zoom_slider_changed(v)

                self.zoom_slider.valueChanged.connect(on_zoom)

    # ── Configuration tab ───────────────────────────────────────────

    def _create_configuration_tab(self) -> QWidget:
        """Configuration tab: engine runtime, simulation options, and Docker image build (the image-build group stays visible regardless of the active runtime)."""
        from src.launchers.runtime_mode_help import (
            make_runtime_mode_help_button,
            show_runtime_mode_help,
        )

        container = QWidget()
        tab_layout = QVBoxLayout(container)

        tab_layout.addWidget(
            self._build_runtime_preferences_group(make_runtime_mode_help_button)
        )
        tab_layout.addWidget(
            self._build_docker_image_group(
                make_runtime_mode_help_button, show_runtime_mode_help
            )
        )

        self._sync_configuration_tab_with_launcher()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(container)

        # DbC postconditions
        assert scroll.widget() is container, (
            "Postcondition: scroll area must wrap the configuration container"
        )
        assert container.layout() is not None, (
            "Postcondition: configuration container must have an active layout"
        )

        return scroll

    def _build_runtime_preferences_group(
        self, make_help_button: Callable[..., QPushButton]
    ) -> QGroupBox:
        """Build the "Runtime & Preferences" group: engine runtime, sim opts, sidekick (renamed from "Execution Environment" to make clear it's the engines' runtime, not the launcher's)."""
        pref_group = QGroupBox("Runtime & Preferences")
        pref_layout = QHBoxLayout(pref_group)

        col_runtime = self._build_engine_runtime_column(make_help_button)
        col_sim = self._build_simulation_options_column()
        col_sidekick = self._build_sidekick_column()

        # Helper function to create divider line
        def make_v_divider() -> QFrame:
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.VLine)
            frame.setFrameShadow(QFrame.Shadow.Sunken)
            frame.setStyleSheet("color: #3a3a3a;")
            return frame

        # Add to main horizontal preferences layout
        pref_layout.addLayout(col_runtime, stretch=1)
        pref_layout.addWidget(make_v_divider())
        pref_layout.addLayout(col_sim, stretch=1)
        pref_layout.addWidget(make_v_divider())
        pref_layout.addLayout(col_sidekick, stretch=1)

        return pref_group

    def _build_engine_runtime_column(
        self, make_help_button: Callable[..., QPushButton]
    ) -> QVBoxLayout:
        """Build the "Engine Runtime" column: Windows/Docker/WSL selection."""
        col_runtime = QVBoxLayout()
        runtime_header = QHBoxLayout()
        lbl_runtime = QLabel("<b>Engine Runtime</b> (pick one):")
        lbl_runtime.setToolTip("Select where physics engines execute.")
        runtime_header.addWidget(lbl_runtime)
        runtime_header.addWidget(make_help_button(self))
        runtime_header.addStretch()
        col_runtime.addLayout(runtime_header)

        grid_runtime = QGridLayout()
        grid_runtime.setSpacing(6)
        grid_runtime.setContentsMargins(0, 0, 0, 0)

        # Row 0: Windows
        self.chk_windows = QCheckBox("Windows")
        self.chk_windows.setToolTip(
            "Run physics engines natively on your local Windows system."
        )
        btn_check_win = QPushButton("Check Deps")
        btn_check_win.setToolTip("Check Windows host environment dependencies")
        btn_check_win.setFixedWidth(100)
        btn_check_win.clicked.connect(self._check_windows_deps)
        grid_runtime.addWidget(self.chk_windows, 0, 0)
        grid_runtime.addWidget(btn_check_win, 0, 1)

        # Row 1: Docker
        self.chk_docker = QCheckBox("Docker")
        self.chk_docker.setToolTip(
            "Run engines inside the upstream-drift:engine Linux container. "
            "Full Drake/Pinocchio support; requires Docker installed and the "
            "image built (see Docker Image section below)."
        )
        self.btn_check_docker_deps = QPushButton("Check Deps")
        self.btn_check_docker_deps.setToolTip(
            "Check Docker container image dependencies"
        )
        self.btn_check_docker_deps.setFixedWidth(100)
        self.btn_check_docker_deps.clicked.connect(self._check_docker_deps)
        grid_runtime.addWidget(self.chk_docker, 1, 0)
        grid_runtime.addWidget(self.btn_check_docker_deps, 1, 1)

        # Row 2: WSL
        self.chk_wsl = QCheckBox("WSL")
        self.chk_wsl.setToolTip(
            "Run engines in your WSL2 Ubuntu user environment. Same Linux "
            "wheels as Docker mode but no container layer — faster file I/O "
            "and easier interactive debugging from a WSL shell."
        )
        self.btn_check_wsl_deps = QPushButton("Check Deps")
        self.btn_check_wsl_deps.setToolTip("Check WSL environment dependencies")
        self.btn_check_wsl_deps.setFixedWidth(100)
        self.btn_check_wsl_deps.clicked.connect(self._check_wsl_deps)
        grid_runtime.addWidget(self.chk_wsl, 2, 0)
        grid_runtime.addWidget(self.btn_check_wsl_deps, 2, 1)

        # Row 3: WSL Setup Script (separate row)
        btn_script_wsl = QPushButton("WSL Setup Script")
        btn_script_wsl.setToolTip("Inspect and run the WSL installation script")
        btn_script_wsl.setFixedWidth(130)
        btn_script_wsl.clicked.connect(self._show_wsl_setup_dialog)
        grid_runtime.addWidget(btn_script_wsl, 3, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)

        col_runtime.addLayout(grid_runtime)
        col_runtime.addStretch()

        from PyQt6.QtWidgets import QButtonGroup

        self.env_group_buttons = QButtonGroup(self)
        self.env_group_buttons.setExclusive(True)
        self.env_group_buttons.addButton(self.chk_windows)
        self.env_group_buttons.addButton(self.chk_docker)
        self.env_group_buttons.addButton(self.chk_wsl)

        return col_runtime

    def _build_simulation_options_column(self) -> QVBoxLayout:
        """Build the "Simulation Options" column: live viz, GPU acceleration."""
        col_sim = QVBoxLayout()
        lbl_sim = QLabel("<b>Simulation Options</b>:")
        col_sim.addWidget(lbl_sim)

        self.chk_live_viz = QCheckBox("Live visualization")
        self.chk_live_viz.setToolTip(
            "Stream the 3D scene in real time during simulation. Disable "
            "for headless batch runs to save GPU/CPU."
        )
        self.chk_gpu = QCheckBox("GPU acceleration")
        self.chk_gpu.setToolTip(
            "Use the GPU for physics where the engine supports it (MuJoCo "
            "MJX, JAX backends). Falls back to CPU if no compatible GPU is "
            "detected — safe to leave on."
        )
        col_sim.addWidget(self.chk_live_viz)
        col_sim.addWidget(self.chk_gpu)
        col_sim.addStretch()

        return col_sim

    def _build_sidekick_column(self) -> QVBoxLayout:
        """Build the "Sidekick AI Assistant" column: context-sharing toggle."""
        col_sidekick = QVBoxLayout()
        lbl_sidekick = QLabel("<b>Sidekick AI Assistant</b>:")
        col_sidekick.addWidget(lbl_sidekick)

        self.chk_sidekick_context = QCheckBox("Share app state with AI")
        self.chk_sidekick_context.setToolTip(
            "When enabled, the Sidekick AI assistant receives a compact summary "
            "of recent diagnostic events and simulation activity as context. "
            "No file paths, passwords, or API keys are included. "
            "Disable to prevent any app state from reaching the assistant."
        )
        self.chk_sidekick_context.setChecked(
            os.environ.get("UPSTREAMDRIFT_SIDEKICK_CONTEXT", "1") != "0"
        )
        self.chk_sidekick_context.toggled.connect(self._on_sidekick_context_toggled)
        col_sidekick.addWidget(self.chk_sidekick_context)

        sidekick_footer = QLabel(
            "<i>Context is capped at ~4 KB; only recent events sent.</i>"
        )
        sidekick_footer.setTextFormat(Qt.TextFormat.RichText)
        col_sidekick.addWidget(sidekick_footer)
        col_sidekick.addStretch()

        return col_sidekick

    def _build_docker_image_group(
        self,
        make_help_button: Callable[..., QPushButton],
        show_help: Callable[[QWidget], None],
    ) -> QGroupBox:
        """Build the "Docker Image" group: stage picker, build/cancel, console (renamed from "Rebuild Environment" — building the image is independent of the runtime selection above)."""
        build_group = QGroupBox("Docker Image")
        build_inner = QVBoxLayout(build_group)
        build_inner.setContentsMargins(8, 8, 8, 8)

        # Create vertical splitter for resizable parts
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(
            self._build_docker_image_upper_section(make_help_button, show_help)
        )

        # Build Console (Lower part of splitter)
        self.build_console = QTextEdit()
        self.build_console.setReadOnly(True)
        self.build_console.setMinimumHeight(150)
        self.build_console.setStyleSheet(Styles.CONSOLE_BUILD)
        splitter.addWidget(self.build_console)

        # Set default proportions
        splitter.setSizes([220, 200])

        build_inner.addWidget(splitter)

        return build_group

    def _build_docker_image_upper_section(
        self,
        make_help_button: Callable[..., QPushButton],
        show_help: Callable[[QWidget], None],
    ) -> QWidget:
        """Build the upper Docker-Image section: stage, tier details, build buttons."""
        upper_widget = QWidget()
        upper_layout = QVBoxLayout(upper_widget)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.setSpacing(6)

        build_header = QHBoxLayout()
        build_header.addWidget(
            QLabel(
                "Build or rebuild the <b>upstream-drift:engine</b> image. "
                "Independent of the runtime selection above."
            )
        )
        build_header.addStretch()
        build_help = make_help_button(self)
        build_help.setToolTip(
            "What does the Docker image contain? Click for full details."
        )
        # The same shared help dialog covers building too — the help
        # text explains how runtime selection and image build relate.
        build_help.clicked.disconnect()
        build_help.clicked.connect(lambda: show_help(self))
        build_header.addWidget(build_help)
        upper_layout.addLayout(build_header)

        stage_row = QHBoxLayout()
        stage_label = QLabel("Target stage:")
        stage_label.setToolTip(
            "Which Dockerfile target to build. 'all' includes every engine "
            "(largest image, longest build, most compatible). The other "
            "stages build only that engine's deps for faster, leaner images."
        )
        stage_row.addWidget(stage_label)
        self.combo_stage = QComboBox()
        self.combo_stage.addItems(list(DOCKER_STAGES))
        self.combo_stage.setToolTip(stage_label.toolTip())
        stage_row.addWidget(self.combo_stage)
        stage_row.addStretch()
        upper_layout.addLayout(stage_row)

        # Docker setup details panel (re-integrated)
        self.tier_details = QTextBrowser()
        self.tier_details.setStyleSheet(Styles.CONSOLE_DIAGNOSTICS)
        self.tier_details.setMinimumHeight(100)
        self.tier_details.setMaximumHeight(150)
        self.tier_details.setOpenExternalLinks(False)
        upper_layout.addWidget(self.tier_details)

        self.combo_stage.currentTextChanged.connect(self._refresh_tier_details)
        self._refresh_tier_details(self.combo_stage.currentText())

        btn_row = QHBoxLayout()
        self._btn_build = QPushButton("Build Image")
        self._btn_build.setToolTip(
            f"Build the {DOCKER_IMAGE_NAME} image now using the selected "
            "target stage. Streams build output below."
        )
        self._btn_build.clicked.connect(self._start_build)
        btn_row.addWidget(self._btn_build)

        self._btn_cancel_build = QPushButton("Cancel")
        self._btn_cancel_build.setToolTip("Abort the running build.")
        self._btn_cancel_build.setEnabled(False)
        self._btn_cancel_build.clicked.connect(self._cancel_build)
        btn_row.addWidget(self._btn_cancel_build)
        upper_layout.addLayout(btn_row)

        self._build_status = QLabel("")
        upper_layout.addWidget(self._build_status)

        return upper_widget

    def _sync_configuration_tab_with_launcher(self) -> None:
        """Wire the configuration tab's checkboxes to the parent launcher's state."""
        launcher = self._launcher
        if launcher:
            # Sync states from launcher
            if hasattr(launcher, "chk_docker") and hasattr(launcher, "chk_wsl"):
                self.chk_windows.setChecked(
                    not launcher.chk_docker.isChecked()
                    and not launcher.chk_wsl.isChecked()
                )
                self.chk_docker.setChecked(launcher.chk_docker.isChecked())
                self.chk_wsl.setChecked(launcher.chk_wsl.isChecked())
            if hasattr(launcher, "chk_live"):
                self.chk_live_viz.setChecked(launcher.chk_live.isChecked())
            if hasattr(launcher, "chk_gpu"):
                self.chk_gpu.setChecked(launcher.chk_gpu.isChecked())

            # Define toggled callbacks that safely set launcher state
            def on_windows_toggled(checked: bool) -> None:
                if checked:
                    if hasattr(launcher, "chk_docker"):
                        launcher.chk_docker.setChecked(False)
                    if hasattr(launcher, "chk_wsl"):
                        launcher.chk_wsl.setChecked(False)

            def on_docker_toggled(checked: bool) -> None:
                if checked:
                    if hasattr(launcher, "chk_docker"):
                        launcher.chk_docker.setChecked(True)
                    if hasattr(launcher, "chk_wsl"):
                        launcher.chk_wsl.setChecked(False)

            def on_wsl_toggled(checked: bool) -> None:
                if checked:
                    if hasattr(launcher, "chk_docker"):
                        launcher.chk_docker.setChecked(False)
                    if hasattr(launcher, "chk_wsl"):
                        launcher.chk_wsl.setChecked(True)

            def invalidate_diagnostics() -> None:
                self._diagnostics_loaded = False

            self.chk_windows.toggled.connect(on_windows_toggled)
            self.chk_docker.toggled.connect(on_docker_toggled)
            self.chk_wsl.toggled.connect(on_wsl_toggled)
            self.chk_windows.toggled.connect(invalidate_diagnostics)
            self.chk_docker.toggled.connect(invalidate_diagnostics)
            self.chk_wsl.toggled.connect(invalidate_diagnostics)

            if hasattr(launcher, "chk_live"):
                self.chk_live_viz.toggled.connect(launcher.chk_live.setChecked)
            if hasattr(launcher, "chk_gpu"):
                self.chk_gpu.toggled.connect(launcher.chk_gpu.setChecked)

            # Two-way sync from launcher to SettingsWidget
            if hasattr(launcher, "chk_windows"):
                launcher.chk_windows.toggled.connect(self.chk_windows.setChecked)
            if hasattr(launcher, "chk_docker"):
                launcher.chk_docker.toggled.connect(self.chk_docker.setChecked)
            if hasattr(launcher, "chk_wsl"):
                launcher.chk_wsl.toggled.connect(self.chk_wsl.setChecked)
            if hasattr(launcher, "chk_live"):
                launcher.chk_live.toggled.connect(self.chk_live_viz.setChecked)
            if hasattr(launcher, "chk_gpu"):
                launcher.chk_gpu.toggled.connect(self.chk_gpu.setChecked)

    # ── MCP Servers tab ─────────────────────────────────────────────

    def _create_mcp_servers_tab(self) -> QWidget:
        """MCP Servers tab: list, add, disable, remove MCP server configs."""
        from PyQt6.QtWidgets import QVBoxLayout, QWidget

        from src.launchers.mcp_servers_preferences import (  # type: ignore[attr-defined]
            McpServersSection,
        )

        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        self._mcp_section = McpServersSection(parent=tab)
        self._mcp_section.restart_required.connect(self._on_mcp_restart_required)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._mcp_section)
        tab_layout.addWidget(scroll)
        return tab

    def _on_mcp_restart_required(self) -> None:
        """Prompt the user to restart the Sidekick chat after MCP config changes."""
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "Restart Required",
            "MCP server configuration changed.\n\n"
            "Restart the Sidekick chat session for changes to take effect.",
        )

    def _on_reset_layout(self) -> None:
        self.reset_layout_requested.emit()

    def _on_layout_lock_toggled(self, checked: bool) -> None:
        """Update layout lock button caption, tooltip, and edit-tiles button when toggled."""
        self._btn_layout_lock.setText(
            "Layout: Unlocked" if checked else "Layout: Locked"
        )
        self._btn_layout_lock.setToolTip(
            "Layout editing is unlocked. Click to lock layout."
            if checked
            else "Layout is locked. Click to unlock layout for editing."
        )
        self._btn_edit_tiles.setEnabled(checked)
        if checked:
            self._btn_edit_tiles.setToolTip("Open dialog to show or hide tiles")
        else:
            self._btn_edit_tiles.setToolTip(
                "Unlock the layout above to edit which tiles are visible"
            )

    def _sync_layout_lock_state(self, checked: bool) -> None:
        """Sync layout lock button state from external action without feedback loops."""
        if self._btn_layout_lock.isChecked() != checked:
            self._btn_layout_lock.setChecked(checked)
        self._on_layout_lock_toggled(checked)

    def _refresh_tier_details(self, profile_name: str) -> None:
        """Show packages and feature list for the selected Docker tier.

        Mirrors the build dialog's tier-details panel — same data source so
        the two views stay in lock-step. Surfaces:

        * Plain-English description of the tier.
        * Max image-size budget vs. estimated installed size.
        * Every feature included (after walking the ``extends:`` chain in
          ``docker/profiles.yaml``) with its display name, approximate size,
          and one-line description.
        """
        info = getattr(self, "_docker_profile_infos", {}).get(profile_name)
        if info is None:
            self.tier_details.setHtml(
                f"<i>No metadata available for profile "
                f"<b>{profile_name}</b>. See <code>docker/profiles.yaml</code>.</i>"
            )
            return

        title = profile_name.replace("-", " ").title()
        rows: list[str] = [f"<b>{title}</b>"]
        if info.description:
            rows.append(f'<span style="color: palette(mid);">{info.description}</span>')
        rows.append("")
        if info.max_size_mb:
            rows.append(
                f"<b>Budget:</b> &le; {info.max_size_mb} MB &nbsp;·&nbsp; "
                f"<b>Estimated install:</b> ~{info.approx_total_mb} MB"
            )
        if info.features:
            rows.append(f"<b>Includes {len(info.features)} feature(s):</b>")
            items: list[str] = []
            for f in info.features:
                size = f"{f.approx_size_mb} MB" if f.approx_size_mb else "—"
                items.append(
                    f"<li><b>{f.display_name}</b> "
                    f'<span style="color: palette(mid);">({size}) — '
                    f"{f.description}</span></li>"
                )
            rows.append("<ul style='margin-top: 4px;'>" + "".join(items) + "</ul>")
        elif info.feature_names:
            rows.append("<b>Features:</b> " + ", ".join(info.feature_names))

        self.tier_details.setHtml("<br>".join(rows))

    def _start_build(self) -> None:
        self.build_console.clear()
        self._btn_build.setEnabled(False)
        self._btn_cancel_build.setEnabled(True)
        self._build_start_time = time.monotonic()
        self._build_timer_id = self.startTimer(1000)
        self._build_status.setText("Building...")

        stage = self.combo_stage.currentText()
        from src.launchers.launcher_constants import DOCKER_STAGES

        if stage in DOCKER_STAGES:
            context = REPOS_ROOT
            dockerfile = REPOS_ROOT / "Dockerfile.modular"
            build_args = {"PROFILE": stage}
        else:
            context = REPOS_ROOT / "src" / "engines" / "physics_engines" / "mujoco"
            dockerfile = None
            build_args = None

        self.build_thread = DockerBuildThread(
            target_stage=stage,
            image_name=DOCKER_IMAGE_NAME,
            context_path=context,
            dockerfile_path=dockerfile,
            build_args=build_args,
        )
        self.build_thread.log_signal.connect(self._on_build_log)
        self.build_thread.finished_signal.connect(self._on_build_finished)
        self.build_thread.start()

    def _on_build_log(self, line: str) -> None:
        if line is None:
            raise ValueError("line must be provided")
        self.build_console.append(line)
        sb = self.build_console.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _on_build_finished(self, success: bool, message: str) -> None:
        if success is None:
            raise ValueError("success must be provided")
        self._btn_build.setEnabled(True)
        self._btn_cancel_build.setEnabled(False)
        if hasattr(self, "_build_timer_id") and self._build_timer_id is not None:
            self.killTimer(self._build_timer_id)
            self._build_timer_id = None
        elapsed = time.monotonic() - self._build_start_time
        status = "SUCCESS" if success else "FAILED"
        self._build_status.setText(f"Build {status} ({elapsed:.0f}s): {message}")
        self.build_console.append(f"\n=== Build {status} ({elapsed:.0f}s) ===")

        if success and self._launcher:
            self._launcher.docker_available = True
            apply_status = getattr(self._launcher, "_apply_docker_status", None)
            if callable(apply_status):
                apply_status(True)
            chk_docker = getattr(self._launcher, "chk_docker", None)
            if chk_docker is not None and hasattr(chk_docker, "setChecked"):
                chk_docker.setChecked(True)

    def _on_sidekick_context_toggled(self, enabled: bool) -> None:
        """Persist the Sidekick context-sharing toggle to the environment.

        Sets ``UPSTREAMDRIFT_SIDEKICK_CONTEXT`` to ``"0"`` when disabled so
        the chat WebSocket skips context injection for this process lifetime.
        The setting is not persisted across process restarts — users must
        re-toggle it in the next session.

        Args:
            enabled: ``True`` to enable context sharing, ``False`` to disable.
        """
        if enabled:
            os.environ.pop("UPSTREAMDRIFT_SIDEKICK_CONTEXT", None)
        else:
            os.environ["UPSTREAMDRIFT_SIDEKICK_CONTEXT"] = "0"
        logger.debug(
            "Sidekick context sharing: %s", "enabled" if enabled else "disabled"
        )

    def _cancel_build(self) -> None:
        if (
            hasattr(self, "build_thread")
            and self.build_thread
            and self.build_thread.isRunning()
        ):
            self._build_status.setText("Cancelling build...")
            self.build_thread.cancel()

    def closeEvent(self, event: Any) -> None:
        """Guard an active Docker build and unsaved preferences (#8895/#8896)."""
        if not self.confirm_close(event):
            return
        super().closeEvent(event)

    def timerEvent(self, event: Any) -> None:
        """Update the build elapsed-time label on each timer tick."""
        if hasattr(self, "_build_start_time"):
            elapsed = time.monotonic() - self._build_start_time
            self._build_status.setText(f"Building... ({elapsed:.0f}s elapsed)")

    def _compare_versions(self, installed: str, required_spec: str) -> bool:
        """Compare installed version string with required specification (e.g., '>=1.26.4').

        Returns True if the requirement is satisfied.
        """
        return _compare_version_strings(installed, required_spec)

    def _generate_dep_table_html(
        self, title: str, env_name: str, check_results: list[dict[str, Any]]
    ) -> str:
        """Generate a beautifully styled HTML table for dependency verification."""
        html = f"""
        <div style="font-family: sans-serif; color: #e2e8f0; background-color: #0f172a; padding: 10px; border-radius: 6px;">
            <h3 style="color: #38bdf8; margin-top: 0; margin-bottom: 8px;">{title}</h3>
            <p style="font-size: 12px; color: #94a3b8; margin-bottom: 12px;">
                Checking installed dependencies for <b>{env_name}</b> runtime:
            </p>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 10px;">
                <thead>
                    <tr style="border-bottom: 2px solid #334155; color: #f8fafc; font-size: 12px; text-align: left;">
                        <th style="padding: 6px 8px;">Package</th>
                        <th style="padding: 6px 8px;">Required</th>
                        <th style="padding: 6px 8px;">Installed</th>
                        <th style="padding: 6px 8px; text-align: center;">Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        for res in check_results:
            name = res["name"]
            req = res["required"]
            inst = res["installed"]
            status = res["status"]

            if status == "ok":
                status_html = (
                    '<span style="color: #4ade80; font-weight: bold;">✅ Pass</span>'
                )
                bg_color = "transparent"
            elif status == "warn":
                status_html = (
                    '<span style="color: #fbbf24; font-weight: bold;">⚠️ Warning</span>'
                )
                bg_color = "rgba(251, 191, 36, 0.05)"
            else:
                status_html = (
                    '<span style="color: #f87171; font-weight: bold;">❌ Fail</span>'
                )
                bg_color = "rgba(248, 113, 113, 0.05)"

            html += f"""
                <tr style="border-bottom: 1px solid #1e293b; background-color: {bg_color}; font-size: 12px;">
                    <td style="padding: 6px 8px; color: #f1f5f9;"><b>{name}</b></td>
                    <td style="padding: 6px 8px; color: #94a3b8;"><code>{req}</code></td>
                    <td style="padding: 6px 8px; color: #cbd5e1;"><code>{inst}</code></td>
                    <td style="padding: 6px 8px; text-align: center;">{status_html}</td>
                </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """
        return html

    def _check_windows_deps(self) -> None:
        """Check the status of native Windows host packages against requirements."""
        from PyQt6.QtWidgets import QMessageBox
        import importlib.metadata
        import sys

        deps = [
            ("NumPy", "numpy", ">=1.26.4", False),
            ("SciPy", "scipy", ">=1.13.1", False),
            ("MuJoCo", "mujoco", ">=3.6.0", False),
            ("PyQt6", "PyQt6", ">=6.5.0", False),
            ("Matplotlib", "matplotlib", ">=3.10.8", False),
            ("Pandas", "pandas", ">=2.0.0", False),
            ("Drake", "pydrake", ">=1.22.0", True),
            ("Pinocchio", "pinocchio", ">=2.6.0", True),
            ("OpenSim", "opensim", ">=4.4.0", True),
            ("MyoSuite", "myosuite", ">=2.0.0", True),
        ]

        check_results = []
        for name, import_name, req, is_opt in deps:
            try:
                __import__(import_name)
                try:
                    v = importlib.metadata.version(import_name)
                except importlib.metadata.PackageNotFoundError:
                    mod = sys.modules.get(import_name)
                    v = getattr(mod, "__version__", None) or getattr(
                        mod, "version", "Unknown"
                    )

                is_ok = self._compare_versions(v, req)
                status = "ok" if is_ok else "error"
                check_results.append(
                    {"name": name, "required": req, "installed": v, "status": status}
                )
            except ImportError:
                if is_opt:
                    check_results.append(
                        {
                            "name": name,
                            "required": req,
                            "installed": "Missing (Use Docker/WSL)",
                            "status": "warn",
                        }
                    )
                else:
                    check_results.append(
                        {
                            "name": name,
                            "required": req,
                            "installed": "Missing",
                            "status": "error",
                        }
                    )

        html = self._generate_dep_table_html(
            "Windows Environment", "Native Windows", check_results
        )
        QMessageBox.information(self, "Windows Dependency Check", html)

    def _check_docker_deps(self) -> None:
        """Verify the built status of the Docker environment container and check package versions inside it."""
        self._start_runtime_dependency_check(
            worker_key="docker",
            button=self.btn_check_docker_deps,
            check_fn=_check_docker_dependencies_report,
        )

    def _check_wsl_deps(self) -> None:
        """Verify python dependency statuses inside the WSL2 environment distro."""
        self._start_runtime_dependency_check(
            worker_key="wsl",
            button=self.btn_check_wsl_deps,
            check_fn=_check_wsl_dependencies_report,
        )

    def _start_runtime_dependency_check(
        self,
        *,
        worker_key: str,
        button: QPushButton,
        check_fn: Callable[[], RuntimeDependencyReport],
    ) -> None:
        """Run a runtime dependency check on a worker thread."""
        worker = self._dep_check_workers.get(worker_key)
        if worker is not None and worker.isRunning():
            return

        original_text = button.text()
        button.setEnabled(False)
        button.setText("Checking...")

        worker = RuntimeDependencyCheckWorker(check_fn, self)
        self._dep_check_workers[worker_key] = worker
        worker.succeeded.connect(
            lambda report, key=worker_key, btn=button, text=original_text: (
                self._on_runtime_dependency_check_succeeded(key, btn, text, report)
            )
        )
        worker.failed.connect(
            functools.partial(
                self._on_runtime_dependency_check_failed,
                worker_key,
                button,
                original_text,
            )
        )
        worker.finished.connect(
            lambda key=worker_key: self._dep_check_workers.pop(key, None)
        )
        worker.start()

    def _restore_dependency_check_button(
        self, worker_key: str, button: QPushButton, original_text: str
    ) -> None:
        self._dep_check_workers.pop(worker_key, None)
        button.setText(original_text)
        button.setEnabled(True)

    def _on_runtime_dependency_check_succeeded(
        self,
        worker_key: str,
        button: QPushButton,
        original_text: str,
        report: RuntimeDependencyReport,
    ) -> None:
        from PyQt6.QtWidgets import QMessageBox

        self._restore_dependency_check_button(worker_key, button, original_text)
        html = self._generate_dep_table_html(
            report.table_title, report.environment_name, report.check_results
        )
        QMessageBox.information(self, report.dialog_title, html)

    def _on_runtime_dependency_check_failed(
        self,
        worker_key: str,
        button: QPushButton,
        original_text: str,
        severity: str,
        title: str,
        html: str,
    ) -> None:
        from PyQt6.QtWidgets import QMessageBox

        self._restore_dependency_check_button(worker_key, button, original_text)
        if severity == "critical":
            QMessageBox.critical(self, title, html)
        else:
            QMessageBox.warning(self, title, html)

    def _show_wsl_setup_dialog(self) -> None:
        """Show the WSL dependency setup script dialog."""
        dialog = WslScriptDialog(self)
        dialog.exec()

    def _copy_wsl_setup_cmd(self) -> None:
        """Copy the bash command to run the WSL dependencies installation script."""
        from PyQt6.QtGui import QGuiApplication
        from PyQt6.QtWidgets import QMessageBox

        cmd = "bash scripts/install_wsl_dependencies.sh"
        clipboard = QGuiApplication.clipboard()
        if clipboard:
            clipboard.setText(cmd)
            QMessageBox.information(
                self,
                "WSL Setup Command Copied",
                "<p>The WSL setup command has been copied to your clipboard:</p>"
                f"<code>{cmd}</code>"
                "<p>Open your WSL terminal in the repository root and run this command to install all Linux-based engine dependencies.</p>",
            )
        else:
            QMessageBox.warning(
                self,
                "Clipboard Error",
                "<p>Could not access the system clipboard. The command is:</p>"
                f"<code>{cmd}</code>",
            )


class SettingsDialog(QDialog):
    """Backward-compatible dialog wrapper around the embedded SettingsWidget."""

    reset_layout_requested = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        diagnostics_data: dict[str, Any] | None = None,
        initial_tab: int = 0,
        launcher: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(850, 650)

        layout = QVBoxLayout(self)
        self.widget = SettingsWidget(
            parent=self,
            diagnostics_data=diagnostics_data,
            initial_tab=initial_tab,
            launcher=launcher if launcher is not None else parent,
        )
        self.widget.reset_layout_requested.connect(self.reset_layout_requested.emit)
        layout.addWidget(self.widget)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        # MUST be close(), not reject(): reject() calls done(), which hides
        # the dialog without dispatching a QCloseEvent, so neither guard ran
        # (#8895/#8896). See settings_close_contract.
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def closeEvent(self, event: Any) -> None:
        """Delegate both close guards to the inner widget (#8896)."""
        if not self.widget.confirm_close(event):
            return
        super().closeEvent(event)

    @property
    def tabs(self) -> Any:
        """Expose the inner tab widget for legacy tests and integrations."""
        return self.widget.tabs
