"""Settings dialog for the UpstreamDrift Launcher.

Provides a tabbed dialog with Layout, Configuration, and Diagnostics tabs.
"""

# mypy: disable-error-code="attr-defined,assignment,union-attr"

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.launchers.docker_manager import DockerBuildThread
from src.launchers.launcher_constants import DOCKER_STAGES
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
    }
    if tab_index not in valid_indexes:
        raise ValueError(
            f"Invalid tab index {tab_index}; expected one of {sorted(valid_indexes)}"
        )
    return tab_index


class SettingsWidget(QWidget):
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
        self._launcher = launcher
        self._diagnostics_loaded = False
        self._setup_ui()
        self.tabs.setCurrentIndex(validate_tab_index(initial_tab))
        # Connect tab change signal for lazy diagnostics loading
        self.tabs.currentChanged.connect(self._on_tab_changed)

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
            }
        )
        self.tabs.setTabsClosable(False)
        layout.addWidget(self.tabs)

        self.tabs.addTab(self._create_layout_tab(), "Layout")
        self.tabs.addTab(self._create_configuration_tab(), "Configuration")
        self.tabs.addTab(self._create_diagnostics_tab(), "Diagnostics")
        self.tabs.addTab(self._create_mcp_servers_tab(), "MCP Servers")

        # Load preferences dialog tabs
        try:
            from src.shared.python.ui.preferences_dialog import PreferencesDialog

            prefs_parent = (
                self._launcher if isinstance(self._launcher, QMainWindow) else None
            )
            self._prefs_dialog = PreferencesDialog(prefs_parent)
            self.tabs.addTab(self._prefs_dialog._create_appearance_tab(), "Appearance")
            self.tabs.addTab(self._prefs_dialog._create_startup_tab(), "Startup")
            self.tabs.addTab(
                self._prefs_dialog._create_notifications_tab(), "Notifications"
            )
            self.tabs.addTab(
                self._prefs_dialog._create_performance_tab(), "Performance"
            )

            # Add an Apply Preferences button
            btn_apply = QPushButton("Apply Preferences")
            btn_apply.clicked.connect(self._prefs_dialog._on_apply)
            layout.addWidget(btn_apply)
        except ImportError:
            pass

    # ── Layout tab ──────────────────────────────────────────────────

    def _create_layout_tab(self) -> QWidget:
        """Layout tab: tile lock, edit tiles, reset to defaults."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        group = QGroupBox("Tile Layout")
        inner = QVBoxLayout(group)

        self._btn_layout_lock = QPushButton("Layout: Locked")
        self._btn_layout_lock.setCheckable(True)
        self._btn_layout_lock.setChecked(False)
        self._btn_layout_lock.setStyleSheet(Styles.BTN_LAYOUT_TOGGLE)
        inner.addWidget(self._btn_layout_lock)

        self._btn_edit_tiles = QPushButton("Edit Tiles (show/hide)")
        self._btn_edit_tiles.setEnabled(False)
        inner.addWidget(self._btn_edit_tiles)

        inner.addSpacing(12)

        btn_reset = QPushButton("Reset Layout to Defaults")
        btn_reset.setToolTip("Restore all tiles and default arrangement")
        btn_reset.clicked.connect(self._on_reset_layout)
        inner.addWidget(btn_reset)

        tab_layout.addWidget(group)

        # --- View Mode --------------------------------------------------
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
        tab_layout.addWidget(view_mode_group)

        # --- Tile Zoom --------------------------------------------------
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
        tab_layout.addWidget(zoom_group)

        # Sync with parent launcher
        launcher = self.parent()
        if launcher and hasattr(launcher, "btn_modify_layout"):
            self._btn_layout_lock.setChecked(launcher.btn_modify_layout.isChecked())
            self._btn_layout_lock.toggled.connect(launcher.btn_modify_layout.click)
            self._btn_edit_tiles.clicked.connect(launcher.open_layout_manager)
            self._btn_layout_lock.toggled.connect(self._btn_edit_tiles.setEnabled)

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

        tab_layout.addStretch()
        return tab

    # ── Configuration tab ───────────────────────────────────────────

    def _create_configuration_tab(self) -> QWidget:
        """Configuration tab: engine runtime + simulation opts + Docker image build.

        The three groups answer three different user questions:

        * **Engine Runtime** — *where do physics engines actually run?*
          (Native Windows / Docker container / WSL2). Inline ``?`` button
          opens the shared help dialog with full pros/cons.
        * **Simulation Options** — per-run knobs (live viz, GPU).
        * **Docker Image** — *build* the container image used by the
          Docker runtime. This group is always visible regardless of
          the active runtime; it's the only place an image build makes
          sense, and the runtime selection is independent.
        """
        from src.launchers.runtime_mode_help import (
            make_runtime_mode_help_button,
            show_runtime_mode_help,
        )

        container = QWidget()
        tab_layout = QVBoxLayout(container)

        # --- Engine Runtime ---------------------------------------------
        # Renamed from "Execution Environment" — "Engine Runtime" makes
        # explicit that this controls where *engines* run, separate from
        # where the launcher itself runs (always Windows).
        # --- Preferences & Engine Runtime --------------------------------
        pref_group = QGroupBox("Runtime & Preferences")
        pref_layout = QHBoxLayout(pref_group)

        # Column 1: Engine Runtime
        col_runtime = QVBoxLayout()
        runtime_header = QHBoxLayout()
        lbl_runtime = QLabel("<b>Engine Runtime</b> (pick one):")
        lbl_runtime.setToolTip("Select where physics engines execute.")
        runtime_header.addWidget(lbl_runtime)
        runtime_header.addWidget(make_runtime_mode_help_button(self))
        runtime_header.addStretch()
        col_runtime.addLayout(runtime_header)

        self.chk_windows = QCheckBox("Windows")
        self.chk_windows.setToolTip(
            "Run physics engines natively on your local Windows system."
        )
        self.chk_docker = QCheckBox("Docker")
        self.chk_docker.setToolTip(
            "Run physics engines inside a Docker container (Linux, sandboxed)."
        )
        self.chk_wsl = QCheckBox("WSL")
        self.chk_wsl.setToolTip(
            "Run physics engines inside WSL2 Ubuntu (Linux, native filesystem)."
        )

        col_runtime.addWidget(self.chk_windows)
        col_runtime.addWidget(self.chk_docker)
        col_runtime.addWidget(self.chk_wsl)
        col_runtime.addStretch()

        from PyQt6.QtWidgets import QButtonGroup

        self.env_group_buttons = QButtonGroup(self)
        self.env_group_buttons.setExclusive(True)
        self.env_group_buttons.addButton(self.chk_windows)
        self.env_group_buttons.addButton(self.chk_docker)
        self.env_group_buttons.addButton(self.chk_wsl)

        # Column 2: Simulation Options
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

        # Column 3: Sidekick AI Assistant
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

        tab_layout.addWidget(pref_group)

        # --- Docker Image build -----------------------------------------
        # Renamed from "Rebuild Environment" — that label conflated the
        # runtime selection with the image build, which are independent.
        # Building the image just puts upstream-drift:engine into your
        # local image store; *using* it requires ticking Docker above.
        build_group = QGroupBox("Docker Image")
        build_inner = QVBoxLayout(build_group)

        build_header = QHBoxLayout()
        build_header.addWidget(
            QLabel(
                "Build or rebuild the <b>upstream-drift:engine</b> image. "
                "Independent of the runtime selection above."
            )
        )
        build_header.addStretch()
        build_help = make_runtime_mode_help_button(self)
        build_help.setToolTip(
            "What does the Docker image contain? Click for full details."
        )
        # The same shared help dialog covers building too — the help
        # text explains how runtime selection and image build relate.
        build_help.clicked.disconnect()
        build_help.clicked.connect(lambda: show_runtime_mode_help(self))
        build_header.addWidget(build_help)
        build_inner.addLayout(build_header)

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
        build_inner.addLayout(stage_row)

        # Tier details: explicit list of what will be installed + total size,
        # so users picking a stage know *exactly* which features they get.
        # Same data source as the Manage Environment dialog
        # (docker/profiles.yaml + feature_registry/features.py).
        from src.launchers.docker_profile_info import (
            ProfileInfo,
            format_profile_summary,
            load_docker_profiles,
        )

        self._docker_profile_infos: dict[str, ProfileInfo] = load_docker_profiles()
        for idx in range(self.combo_stage.count()):
            info = self._docker_profile_infos.get(self.combo_stage.itemText(idx))
            if info is not None:
                self.combo_stage.setItemData(
                    idx,
                    format_profile_summary(info),
                    Qt.ItemDataRole.ToolTipRole,
                )

        # ``QTextBrowser`` instead of ``QLabel-in-QScrollArea`` because the
        # latter clipped content rather than scrolling when paired with
        # ``widgetResizable=True`` (Qt sizes the label to the viewport, so
        # the rich-text rendering has nowhere to spill, and the scrollbar
        # handle never appears). QTextBrowser is purpose-built for read-only
        # rich text with native scrolling and selection.
        from PyQt6.QtWidgets import QSizePolicy as _QSizePolicy

        self.tier_details = QTextBrowser()
        self.tier_details.setReadOnly(True)
        self.tier_details.setOpenExternalLinks(False)
        self.tier_details.setFrameShape(QFrame.Shape.StyledPanel)
        self.tier_details.setFixedHeight(200)
        self.tier_details.setSizePolicy(
            _QSizePolicy.Policy.Expanding, _QSizePolicy.Policy.Fixed
        )
        self.tier_details.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.tier_details.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        build_inner.addWidget(self.tier_details)

        self.combo_stage.currentTextChanged.connect(self._refresh_tier_details)
        self._refresh_tier_details(self.combo_stage.currentText())

        # Visible gap between the tier-details panel and the action buttons
        # below so a long features list can't visually bleed into them.
        build_inner.addSpacing(12)

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
        build_inner.addLayout(btn_row)

        self._build_status = QLabel("")
        build_inner.addWidget(self._build_status)

        self.build_console = QTextEdit()
        self.build_console.setReadOnly(True)
        self.build_console.setMaximumHeight(150)
        self.build_console.setStyleSheet(Styles.CONSOLE_BUILD)
        build_inner.addWidget(self.build_console)

        tab_layout.addWidget(build_group)

        # Sync checkboxes with parent launcher state
        launcher = self._launcher or self.parent()
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

    # ── Diagnostics tab ─────────────────────────────────────────────

    def _create_diagnostics_tab(self) -> QWidget:
        """Diagnostics tab: system checks, error log viewer, terminal output."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # System checks browser
        self._diag_browser = QTextBrowser()
        self._diag_browser.setOpenExternalLinks(False)
        self._diag_browser.setStyleSheet(Styles.CONSOLE_DIAGNOSTICS)
        tab_layout.addWidget(self._diag_browser, stretch=3)

        if self._diagnostics_data:
            self._render_diagnostics(self._diagnostics_data)

        # Process output log viewer
        proc_group = QGroupBox("Process Output Log (recent)")
        proc_inner = QVBoxLayout(proc_group)
        self._proc_log_viewer = QTextEdit()
        self._proc_log_viewer.setReadOnly(True)
        self._proc_log_viewer.setMaximumHeight(180)
        self._proc_log_viewer.setStyleSheet(Styles.CONSOLE_LOG_GREEN)
        proc_inner.addWidget(self._proc_log_viewer)
        tab_layout.addWidget(proc_group, stretch=1)

        # Application log viewer
        log_group = QGroupBox("Application Log (recent)")
        log_inner = QVBoxLayout(log_group)
        self._log_viewer = QTextEdit()
        self._log_viewer.setReadOnly(True)
        self._log_viewer.setMaximumHeight(160)
        self._log_viewer.setStyleSheet(Styles.CONSOLE_LOG_LIGHT)
        log_inner.addWidget(self._log_viewer)
        tab_layout.addWidget(log_group, stretch=1)

        # Load recent log lines
        self._load_process_log()
        self._load_app_log()

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.btn_sync_tools = QPushButton("Sync Shared Tools")
        self.btn_sync_tools.setToolTip(
            "Synchronize git submodules and fetch latest updates"
        )
        self.btn_sync_tools.clicked.connect(self._sync_shared_tools)
        btn_row.addWidget(self.btn_sync_tools)

        btn_refresh = QPushButton("Re-run Diagnostics")
        btn_refresh.setToolTip("Run all diagnostic checks again")
        btn_refresh.clicked.connect(self._refresh_diagnostics)
        btn_row.addWidget(btn_refresh)

        btn_refresh_log = QPushButton("Refresh Logs")
        btn_refresh_log.setToolTip("Reload all log files")
        btn_refresh_log.clicked.connect(self._refresh_all_logs)
        btn_row.addWidget(btn_refresh_log)

        tab_layout.addLayout(btn_row)
        return tab

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

    def _load_app_log(self) -> None:
        """Load recent lines from the application log file."""
        log_candidates = [
            Path.cwd() / "app_launch.log",
            Path.home() / ".golf_modeling_suite" / "launcher.log",
        ]
        for log_path in log_candidates:
            if log_path.exists():
                try:
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                    lines = text.strip().splitlines()
                    recent = "\n".join(lines[-200:])
                    self._log_viewer.setPlainText(recent)
                    from PyQt6.QtGui import QTextCursor

                    self._log_viewer.moveCursor(QTextCursor.MoveOperation.End)
                    return
                except (RuntimeError, ValueError, AttributeError, OSError) as e:
                    logger.debug("Could not display log file %s: %s", log_path, e)
        self._log_viewer.setPlainText("(No log file found)")

    def _load_process_log(self) -> None:
        """Load recent lines from the process output log file."""
        log_path = Path.home() / ".golf_modeling_suite" / "process_output.log"
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                lines = text.strip().splitlines()
                recent = "\n".join(lines[-300:])
                self._proc_log_viewer.setPlainText(recent)
                from PyQt6.QtGui import QTextCursor

                self._proc_log_viewer.moveCursor(QTextCursor.MoveOperation.End)
                return
            except (RuntimeError, ValueError, AttributeError, OSError) as e:
                logger.debug("Could not display process log %s: %s", log_path, e)
        self._proc_log_viewer.setPlainText(
            "(No process output log yet — launch a model to generate output)"
        )

    def _refresh_all_logs(self) -> None:
        """Refresh both log viewers."""
        self._load_process_log()
        self._load_app_log()

    def _render_diagnostics(self, data: dict[str, Any]) -> None:
        """Render diagnostics results as styled HTML."""
        if data is None:
            raise ValueError("data must be provided")
        summary = data.get("summary", {})
        checks = data.get("checks", [])
        runtime = data.get("runtime_state", {})
        recommendations = data.get("recommendations", [])

        html = self._render_diag_summary(summary)
        html += self._render_diag_checks(checks)
        html += self._render_diag_engines(checks)
        html += self._render_diag_runtime(runtime)
        html += self._render_diag_recommendations(recommendations)

        self._diag_browser.setHtml(html)

    def _render_diag_summary(self, summary: dict) -> str:
        if summary is None:
            raise ValueError("summary must be provided")
        status = summary.get("status", "unknown").upper()
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        warnings = summary.get("warnings", 0)
        total = summary.get("total_checks", passed + failed + warnings)

        status_color = "#2da44e" if status == "HEALTHY" else "#d29922"
        return f"""
        <div style="margin-bottom: 12px;">
            <h2 style="color:{status_color}; margin: 0;">Status: {status}</h2>
            <p><b>{total} checks:</b>
                <span style="color:#2da44e;">{passed} passed</span>,
                <span style="color:#f85149;">{failed} failed</span>,
                <span style="color:#d29922;">{warnings} warnings</span>
            </p>
        </div>
        """

    def _render_diag_checks(self, checks: list) -> str:
        if checks is None:
            raise ValueError("checks must be provided")
        html = "<h3>Check Results</h3><table style='width:100%;'>"
        for check in checks:
            icon = {"pass": "&#9989;", "fail": "&#10060;", "warning": "&#9888;"}.get(
                check["status"], "&#8226;"
            )
            color = {"pass": "#2da44e", "fail": "#f85149", "warning": "#d29922"}.get(
                check["status"], "#d4d4d4"
            )
            duration = check.get("duration_ms", 0)
            html += (
                f"<tr><td style='color:{color}; padding:2px 6px;'>{icon}</td>"
                f"<td style='padding:2px 6px;'><b>{check['name']}</b></td>"
                f"<td style='padding:2px 6px; color:#a0a0a0;'>{check['message']}</td>"
                f"<td style='padding:2px 6px; color:#666;'>{duration:.0f}ms</td></tr>"
            )
        html += "</table>"
        return html

    def _render_diag_engines(self, checks: list) -> str:
        if checks is None:
            raise ValueError("checks must be provided")
        engine_check = next(
            (c for c in checks if c["name"] == "engine_availability"), None
        )
        engines = (
            engine_check.get("details", {}).get("engines", []) if engine_check else []
        )
        if not engines:
            return ""

        html = "<h3>Physics Engines</h3>"
        html += (
            "<table style='width:100%; border-collapse:collapse;'>"
            "<tr style='border-bottom:1px solid #333;'>"
            "<th style='padding:4px 8px; text-align:left;'>Engine</th>"
            "<th style='padding:4px 8px; text-align:left;'>Status</th>"
            "<th style='padding:4px 8px; text-align:left;'>Version</th>"
            "<th style='padding:4px 8px; text-align:left;'>Details</th>"
            "</tr>"
        )
        for eng in engines:
            installed = eng.get("installed", False)
            icon = "&#9989;" if installed else "&#10060;"
            color = "#2da44e" if installed else "#f85149"
            name = eng.get("name", "?").replace("_", " ").title()
            version = eng.get("version") or "-"
            diag = eng.get("diagnostic", "")
            missing = eng.get("missing_deps", [])
            detail_str = diag
            if missing and not installed:
                detail_str = f"Missing: {', '.join(missing[:3])}"
            html += (
                f"<tr>"
                f"<td style='padding:3px 8px;'><b>{name}</b></td>"
                f"<td style='padding:3px 8px; color:{color};'>{icon} "
                f"{'Installed' if installed else 'Not installed'}</td>"
                f"<td style='padding:3px 8px; color:#a0a0a0;'>{version}</td>"
                f"<td style='padding:3px 8px; color:#888;'>{detail_str}</td>"
                f"</tr>"
            )
        html += "</table>"
        return html

    def _render_diag_runtime(self, runtime: dict) -> str:
        if runtime is None:
            raise ValueError("runtime must be provided")
        if not runtime:
            return ""
        html = "<h3>Runtime State</h3><ul>"
        html += (
            f"<li>Available models: {runtime.get('available_models_count', '?')}</li>"
        )
        html += f"<li>Tile order: {runtime.get('model_order_count', '?')}</li>"
        html += f"<li>Model cards: {runtime.get('model_cards_count', '?')}</li>"
        html += f"<li>Registry loaded: {runtime.get('registry_loaded', '?')}</li>"
        html += f"<li>Docker available: {runtime.get('docker_available', '?')}</li>"
        html += "</ul>"
        return html

    def _render_diag_recommendations(self, recommendations: list) -> str:
        if recommendations is None:
            raise ValueError("recommendations must be provided")
        if not recommendations:
            return ""
        html = "<h3>Recommendations</h3><ul>"
        for rec in recommendations[:8]:
            html += f"<li>{rec}</li>"
        html += "</ul>"
        return html

    def _refresh_diagnostics(self) -> None:
        """Re-run diagnostics and update the display."""
        try:
            from src.launchers.launcher_diagnostics import LauncherDiagnostics

            diag = LauncherDiagnostics()
            results = diag.run_all_checks()

            launcher = self.parent()
            if launcher and hasattr(launcher, "available_models"):
                results["runtime_state"] = {
                    "available_models_count": len(launcher.available_models),
                    "available_model_ids": list(launcher.available_models.keys()),
                    "model_order_count": len(launcher.model_order),
                    "model_order": launcher.model_order,
                    "model_cards_count": len(launcher.model_cards),
                    "selected_model": launcher.selected_model,
                    "docker_available": launcher.docker_available,
                    "registry_loaded": launcher.registry is not None,
                }

            self._diagnostics_data = results
            self._render_diagnostics(results)
        except ImportError as e:
            self._diag_browser.setHtml(
                f"<p style='color:#f85149;'>Error running diagnostics: {e}</p>"
            )

    def _on_reset_layout(self) -> None:
        self.reset_layout_requested.emit()

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
            self._launcher._apply_docker_status(True)
            self._launcher.chk_docker.setChecked(True)

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
            self.build_thread.terminate()
            self._build_status.setText("Build cancelled.")
            self._btn_build.setEnabled(True)
            self._btn_cancel_build.setEnabled(False)
            if hasattr(self, "_build_timer_id") and self._build_timer_id is not None:
                self.killTimer(self._build_timer_id)
                self._build_timer_id = None

    def timerEvent(self, event: Any) -> None:
        """Update the build elapsed-time label on each timer tick."""
        if hasattr(self, "_build_start_time"):
            elapsed = time.monotonic() - self._build_start_time
            self._build_status.setText(f"Building... ({elapsed:.0f}s elapsed)")

    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change for lazy diagnostics loading.

        When the Diagnostics tab is first selected, run diagnostics
        asynchronously and populate the display. Subsequent selections
        do not re-run diagnostics.

        Args:
            index: The new tab index.
        """
        if index != TAB_DIAGNOSTICS or self._diagnostics_loaded:
            return

        # Mark as loaded to prevent re-running on subsequent tab visits
        self._diagnostics_loaded = True

        # Show loading indicator
        self._diag_browser.setHtml(
            "<p style='color:#d29922;'>Running diagnostics...</p>"
        )

        # Run diagnostics in a background thread
        from PyQt6.QtCore import QThread, pyqtSignal

        class DiagnosticsWorker(QThread):
            """Background worker for running diagnostics."""

            finished = pyqtSignal(dict)
            error = pyqtSignal(str)

            def __init__(self, launcher: Any | None = None) -> None:
                super().__init__()
                self._launcher = launcher

            def run(self) -> None:
                try:
                    from src.launchers.launcher_diagnostics import LauncherDiagnostics

                    diag = LauncherDiagnostics()
                    results = diag.run_all_checks()

                    if self._launcher and hasattr(self._launcher, "available_models"):
                        results["runtime_state"] = {
                            "available_models_count": len(
                                self._launcher.available_models
                            ),
                            "available_model_ids": list(
                                self._launcher.available_models.keys()
                            ),
                            "model_order_count": len(self._launcher.model_order),
                            "model_order": self._launcher.model_order,
                            "model_cards_count": len(self._launcher.model_cards),
                            "selected_model": self._launcher.selected_model,
                            "docker_available": self._launcher.docker_available,
                            "registry_loaded": self._launcher.registry is not None,
                        }

                    self.finished.emit(results)
                except ImportError as e:
                    self.error.emit(str(e))

        self._diag_worker = DiagnosticsWorker(self._launcher)
        self._diag_worker.finished.connect(self._on_diagnostics_ready)
        self._diag_worker.error.connect(self._on_diagnostics_error)
        self._diag_worker.start()

    def _on_diagnostics_ready(self, results: dict[str, Any]) -> None:
        """Handle completed diagnostics results.

        Args:
            results: The diagnostics results dictionary.
        """
        self._diagnostics_data = results
        self._render_diagnostics(results)

    def _on_diagnostics_error(self, error_msg: str) -> None:
        """Handle diagnostics error.

        Args:
            error_msg: The error message.
        """
        self._diag_browser.setHtml(
            f"<p style='color:#f85149;'>Error running diagnostics: {error_msg}</p>"
        )

    def _sync_shared_tools(self) -> None:
        """Run submodule synchronization and fetch latest remote commits in a background thread."""
        from PyQt6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Sync Shared Tools",
            "Are you sure you want to synchronize git submodules and fetch latest remote updates?\n\n"
            "This will run 'git submodule update --init --recursive' in the background.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            return

        self.btn_sync_tools.setEnabled(False)
        self.btn_sync_tools.setText("Syncing...")
        self._diag_browser.append(
            "<p style='color:#58a6ff;'>Starting sync of shared tools...</p>"
        )

        from PyQt6.QtCore import QThread, pyqtSignal
        from src.shared.python.data_io.path_utils import get_repo_root
        from src.launchers.launcher_diagnostics import LauncherDiagnostics
        import subprocess

        class SubmoduleSyncWorker(QThread):
            finished = pyqtSignal(bool, str)

            def __init__(self, repos_root: Path) -> None:
                super().__init__()
                self.repos_root = repos_root

            def run(self) -> None:
                try:
                    # 1. git submodule update
                    result = subprocess.run(
                        ["git", "submodule", "update", "--init", "--recursive"],
                        cwd=str(self.repos_root),
                        capture_output=True,
                        text=True,
                        check=True,
                        encoding="utf-8",
                        timeout=60.0,
                    )

                    # 2. Fetch updates for parent repo
                    subprocess.run(
                        ["git", "fetch"],
                        cwd=str(self.repos_root),
                        capture_output=True,
                        timeout=15.0,
                    )

                    # 3. Fetch updates for submodule
                    submodule_dir = self.repos_root / "vendor" / "ud-tools"
                    if submodule_dir.is_dir():
                        subprocess.run(
                            ["git", "fetch"],
                            cwd=str(submodule_dir),
                            capture_output=True,
                            timeout=15.0,
                        )

                    # 4. Fetch updates for sibling Tools
                    sibling_root = LauncherDiagnostics._find_sibling_tools_root()
                    if sibling_root:
                        subprocess.run(
                            ["git", "fetch"],
                            cwd=str(sibling_root),
                            capture_output=True,
                            timeout=15.0,
                        )

                    output_msg = (
                        result.stdout or "Submodules synchronized successfully."
                    )
                    self.finished.emit(True, output_msg)
                except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
                    self.finished.emit(False, str(e))

        repos_root = get_repo_root()
        self._sync_worker = SubmoduleSyncWorker(repos_root)
        self._sync_worker.finished.connect(self._on_sync_finished)
        self._sync_worker.start()

    def _on_sync_finished(self, success: bool, output: str) -> None:
        """Handle completion of the synchronization worker thread."""
        self.btn_sync_tools.setEnabled(True)
        self.btn_sync_tools.setText("Sync Shared Tools")

        from PyQt6.QtWidgets import QMessageBox

        if success:
            QMessageBox.information(
                self,
                "Sync Complete",
                "Shared tools synchronization completed successfully.",
            )
            # Re-run diagnostics to update status
            self._refresh_diagnostics()
        else:
            QMessageBox.critical(
                self,
                "Sync Failed",
                f"Failed to synchronize shared tools:\n\n{output}",
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
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def tabs(self) -> Any:
        """Expose the inner tab widget for legacy tests and integrations."""
        return self.widget.tabs
