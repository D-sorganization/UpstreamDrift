"""Dialog and settings management mixin for GolfLauncher.

Contains methods for help dialogs, about dialog, shortcuts overlay,
preferences, settings, diagnostics, environment manager, layout manager,
bug reporting, and AI settings.
"""

# mypy: disable-error-code="attr-defined,call-overload,arg-type,assignment"

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import QMessageBox, QDialog

from src.launchers.launcher_constants import (
    AI_AVAILABLE,
    CREATE_NO_WINDOW,
    HELP_SYSTEM_AVAILABLE,
    REPOS_ROOT,
    UI_COMPONENTS_AVAILABLE,
)
from src.launchers.ui_components import (
    LayoutManagerDialog,
    SettingsDialog,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class LauncherDialogsMixin:
    """Mixin for GolfLauncher dialog and settings management.

    Provides methods for displaying help, about, shortcuts, preferences,
    settings, diagnostics, environment manager, layout manager dialogs,
    and AI settings.
    """

    def _init_ui_components(self) -> None:
        """Initialize optional UI components (toast, shortcuts, etc.)."""
        # Toast notification manager
        if UI_COMPONENTS_AVAILABLE:
            from src.shared.python.ui import ToastManager

            self.toast_manager: ToastManager | None = ToastManager(self)

            # Setup keyboard shortcuts
            self._setup_keyboard_shortcuts()
        else:
            self.toast_manager = None

        # Register the Sidekick feature Window menu (Tools surfacing).
        # Done after the menu bar exists; tolerated as a no-op when
        # menuBar() returns None (e.g. test fixtures without a window).
        self._register_feature_window_menu()

    def _register_feature_window_menu(self) -> None:
        """Add the Window menu surfacing Sidekick features to the menu bar.

        Idempotent: safe to call multiple times (we drop a previously
        attached menu before re-adding).
        """
        menubar = getattr(self, "menu_bar", None)
        if menubar is None:
            return
        try:
            from src.launchers.feature_menu import register_feature_menu

            self._feature_menu_actions = register_feature_menu(self, menubar)
        except ImportError as exc:  # pragma: no cover - guarded
            logger.debug("feature_menu unavailable: %s", exc)
            self._feature_menu_actions = {}

    def _setup_keyboard_shortcuts(self) -> None:
        """Set up global keyboard shortcuts."""
        # F1 for help dialog (User Manual)
        shortcut_f1 = QShortcut(QKeySequence("F1"), self)
        shortcut_f1.activated.connect(self._show_help_dialog)

        # Ctrl+? for shortcuts overlay
        shortcut_help = QShortcut(QKeySequence("Ctrl+?"), self)
        shortcut_help.activated.connect(self._show_shortcuts_overlay)

        # Ctrl+, for preferences
        shortcut_prefs = QShortcut(QKeySequence("Ctrl+,"), self)
        shortcut_prefs.activated.connect(self._show_preferences)

        # Ctrl+Q to quit
        shortcut_quit = QShortcut(QKeySequence("Ctrl+Q"), self)
        shortcut_quit.activated.connect(self.close)

        # Sidekick feature shortcuts (Tools #2882/#2883/#2884/#2888/#2889).
        # The single source of truth lives in feature_menu.FEATURE_ENTRIES so
        # menu actions and these shortcuts cannot drift apart.
        try:
            from src.launchers.feature_menu import FEATURE_ENTRIES

            for entry in FEATURE_ENTRIES:
                if not entry.availability_probe():
                    continue
                sc = QShortcut(QKeySequence(entry.shortcut), self)
                sc.activated.connect(lambda e=entry: e.factory(self))
        except ImportError as exc:  # pragma: no cover — guard import path
            logger.debug("feature_menu not importable: %s", exc)

    def _show_help_dialog(self, topic: str | None = None) -> None:
        """Show the help dialog.

        Args:
            topic: Optional help topic to display initially.
        """
        if HELP_SYSTEM_AVAILABLE:
            from src.shared.python.gui_pkg.help_system import HelpDialog

            dialog = HelpDialog(self, initial_topic=topic)
            dialog.exec()
        else:
            from src.launchers.ui_components import HelpDialog as LegacyHelpDialog

            dialog = LegacyHelpDialog(self)
            dialog.exec()

    def _open_project_map(self) -> None:
        """Open the Project Map document in the system viewer."""
        project_map = REPOS_ROOT / "docs" / "PROJECT_MAP.md"
        if project_map.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(project_map)))
        else:
            QMessageBox.warning(
                self,
                "Project Map Not Found",
                "The Project Map file was not found at:\n"
                f"{project_map}\n\n"
                "Please ensure docs/PROJECT_MAP.md exists.",
            )

    def _show_about_dialog(self) -> None:
        """Show the About dialog."""
        QMessageBox.about(
            self,
            "About UpstreamDrift",
            "<h2>UpstreamDrift</h2>"
            "<h3>Biomechanical Golf Swing Analysis</h3>"
            "<p><b>Version 2.1</b></p>"
            "<p>Biomechanical Golf Swing Analysis Platform</p>"
            "<hr>"
            "<p>A unified platform for biomechanical golf swing analysis "
            "integrating multiple physics engines including MuJoCo, Drake, "
            "Pinocchio, OpenSim, and MyoSuite.</p>"
            "<p>Copyright 2024-2026 UpstreamDrift Contributors</p>"
            '<p><a href="https://github.com/dieterolson/UpstreamDrift">GitHub Repository</a></p>',
        )

    def _show_shortcuts_overlay(self) -> None:
        """Show the keyboard shortcuts overlay."""
        if UI_COMPONENTS_AVAILABLE:
            from src.shared.python.ui import ShortcutsOverlay

            overlay = ShortcutsOverlay(self)
            overlay.show()
            overlay.setFocus()

    def _show_preferences(self) -> None:
        """Show the preferences dialog."""
        if UI_COMPONENTS_AVAILABLE:
            from src.shared.python.ui import PreferencesDialog

            dialog = PreferencesDialog(self)
            dialog.exec()

    def open_sidekick_tab(self, tool_id: str) -> None:
        """Open *tool_id* as a Sidekick tab in the launcher.

        Best-effort dispatcher that delegates to the embedded host if
        available. Logs a warning (and shows a toast) when the host or
        the tool isn't wired up — never raises, so the menu/shortcut
        path remains robust during the transitional period while
        Sidekick tabs are being wired feature-by-feature.

        Tools surfaced through this hook: #2882 (OS terminal),
        #2883 (Python REPL, workspace), #2884 (MCP servers),
        #2888 (skills), #2889 (Jupyter).
        """
        if not tool_id:
            raise ValueError("tool_id must be non-empty")

        host = getattr(self, "embedded_host", None)
        opener = getattr(host, "open_tab", None) if host is not None else None
        if callable(opener):
            try:
                opener(tool_id)
                return
            except Exception as exc:  # noqa: BLE001 — bubble via toast
                logger.warning("embedded_host.open_tab(%r) failed: %s", tool_id, exc)
                self.show_toast(f"Failed to open {tool_id} tab: {exc}", "error")
                return

        logger.info(
            "open_sidekick_tab(%r): embedded host not available yet — "
            "Sidekick tab integration lands with Tools surfacing PR.",
            tool_id,
        )
        self.show_toast(
            f"Sidekick tab '{tool_id}' is not yet wired in this build.",
            "info",
        )

    def open_preferences_section(self, section_id: str) -> None:
        """Open the preferences dialog focused on *section_id*.

        Currently delegates to the generic preferences entry point; the
        section_id parameter is accepted so callers can use a stable
        API while a section-aware dialog is wired up.
        """
        if not section_id:
            raise ValueError("section_id must be non-empty")
        logger.debug("open_preferences_section(%r)", section_id)
        self._show_preferences()

    def show_toast(self, message: str, toast_type: str = "info") -> None:
        """Show a toast notification.

        Args:
            message: Message to display
            toast_type: Type of toast ("success", "error", "warning", "info")
        """
        if self.toast_manager:
            if toast_type == "success":
                self.toast_manager.show_success(message)
            elif toast_type == "error":
                self.toast_manager.show_error(message)
            elif toast_type == "warning":
                self.toast_manager.show_warning(message)
            else:
                self.toast_manager.show_info(message)

    def _open_ai_settings(self) -> None:
        """Open the AI settings dialog."""
        if not AI_AVAILABLE:
            return

        from src.shared.python.ai.gui import AISettingsDialog

        dialog = AISettingsDialog(self)
        # Reload settings in panel
        if dialog.exec() and hasattr(self, "ai_panel"):
            pass

    def _open_integrations_health(self) -> None:
        """Open the integrations health dashboard window (UD #5643).

        Hosts the shared :class:`IntegrationsHealthDashboardWidget` from
        Tools (PR #2914) in a modeless dialog.
        """
        from src.launchers.integrations_health_window import (
            open_integrations_health_window,
        )

        # Keep a reference so the dialog isn't garbage-collected.
        self._integrations_health_dialog = open_integrations_health_window(self)

    def toggle_ai_assistant(self, checked: bool) -> None:
        """Toggle the AI Assistant panel visibility via the content splitter.

        Args:
            checked: Whether the button is checked.
        """
        if checked is None:
            raise ValueError("checked must be provided")
        if not AI_AVAILABLE:
            return

        self._ai_visible = checked
        # Keep the toggle button in sync when called programmatically
        if hasattr(self, "btn_ai_sidebar") and self.btn_ai_sidebar.isChecked() != checked:
            self.btn_ai_sidebar.setChecked(checked)

        if hasattr(self, "sidekick_sidebar") and self.sidekick_sidebar is not None:
            self.sidekick_sidebar.setVisible(checked)
            if checked and hasattr(self, "open_sidekick_tab"):
                self.open_sidekick_tab("chat")

    def _report_bug(self) -> None:
        """Open default mail client to report a bug."""
        subject = "Bug Report: UpstreamDrift"
        body = "Please describe the issue you encountered:\n\n"

        from urllib.parse import quote

        email = "support@golfmodelingsuite.com"
        mailto_url = f"mailto:{email}?subject={quote(subject)}&body={quote(body)}"

        QDesktopServices.openUrl(QUrl(mailto_url))

    def _open_settings(self, tab: int = 0) -> None:
        """Open the Settings dialog (Layout / Configuration / Diagnostics).

        Args:
            tab: Initial tab index. See ``settings_dialog.SettingsDialog``
                tab constants (``TAB_LAYOUT`` / ``TAB_CONFIG`` /
                ``TAB_DIAGNOSTICS``); the Configuration tab is where
                Engine Runtime selection and Docker Image build live.

        Note: Diagnostics are now loaded lazily when the Diagnostics tab
        is first selected, not synchronously before dialog display.
        See issue #4916.
        """
        if tab is None:
            raise ValueError("tab must be provided")

        dialog = SettingsDialog(
            parent=self,
            diagnostics_data=None,  # Lazy-load on tab select
            initial_tab=tab,
            launcher=self,
        )
        dialog.reset_layout_requested.connect(self._reset_layout_to_defaults)
        dialog.exec()

    def open_diagnostics(self) -> None:
        """Open the settings dialog on the Diagnostics tab."""
        self._open_settings(tab=2)

    def open_environment_manager(self) -> None:
        """Open the settings dialog on the Configuration tab."""
        self._open_settings(tab=1)

    def _reset_layout_to_defaults(self) -> None:
        """Reset layout configuration to show all default tiles."""
        config_file = Path.home() / ".golf_modeling_suite" / "launcher_layout.json"

        try:
            if config_file.exists():
                backup_path = config_file.with_suffix(".json.bak")
                config_file.rename(backup_path)
                logger.info(f"Backed up existing config to {backup_path}")

            self._initialize_model_order()
            self._sync_model_cards()
            self._rebuild_grid()

            self.show_toast("Layout reset to defaults", "success")
            logger.info("Layout reset to defaults")

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to reset layout: {e}")
            self.show_toast(f"Failed to reset layout: {e}", "error")

    def open_help(self) -> None:
        """Open the help dialog.

        Note: This method is kept for backward compatibility.
        Use _show_help_dialog() for new code.
        """
        self._show_help_dialog()

    def open_layout_manager(self) -> None:
        """Open the layout customization dialog."""
        dialog = LayoutManagerDialog(self.available_models, self.model_order, self)
        if dialog.exec():
            selected = dialog.selected_ids()
            self._apply_model_selection(selected)
            self.show_toast("Layout updated", "success")

    def toggle_layout_mode(self, checked: bool) -> None:
        """Toggle tile editing mode."""
        if checked is None:
            raise ValueError("checked must be provided")
        self.layout_edit_mode = checked
        self.layout_manager.set_edit_mode(checked)
        if checked:
            self.btn_modify_layout.setText("Layout: Unlocked 🔓")
            self.btn_modify_layout.setStyleSheet(Styles.BTN_LAYOUT_EDIT_ON)
            if hasattr(self, "action_customize_tiles"):
                self.action_customize_tiles.setEnabled(True)
            self.show_toast("Drag tiles to reorder. Double-click to launch.", "info")
        else:
            self.btn_modify_layout.setText("Layout: Locked 🔒")
            self.btn_modify_layout.setStyleSheet(Styles.BTN_LAYOUT_LOCKED)
            if hasattr(self, "action_customize_tiles"):
                self.action_customize_tiles.setEnabled(False)

    def _on_docker_mode_changed(self, state: int) -> None:
        """Handle Docker mode toggle change.

        Args:
            state: Qt checkbox state (0=unchecked, 2=checked)
        """
        if state is None:
            raise ValueError("state must be provided")
        use_docker = state == 2
        if use_docker:
            # Disable WSL mode if Docker is enabled (mutually exclusive)
            if hasattr(self, "chk_wsl") and self.chk_wsl.isChecked():
                self.chk_wsl.setChecked(False)

            if not self.docker_available:
                QMessageBox.warning(
                    self,
                    "Docker Not Available",
                    "Docker Desktop is not running or not installed.\n\n"
                    "Please start Docker Desktop and try again.\n\n"
                    "The launcher will continue in local mode.",
                )
                self.chk_docker.setChecked(False)
                return

        if use_docker:
            logger.info("Docker mode enabled")
            if hasattr(self, "toast_manager") and self.toast_manager:
                self.show_toast(
                    "Docker mode enabled - engines will run in containers", "info"
                )
        else:
            logger.info("Docker mode disabled")
            if hasattr(self, "toast_manager") and self.toast_manager:
                self.show_toast("Local mode - engines will run on host system", "info")

        # Update UI status
        self.update_execution_status()

        # Update launch button text if a model is selected
        if hasattr(self, "btn_launch"):
            self.update_launch_button()

    def _on_wsl_mode_changed(self, state: int) -> None:  # noqa: C901
        """Handle WSL mode toggle change.

        Args:
            state: Qt checkbox state (0=unchecked, 2=checked)
        """
        if state is None:
            raise ValueError("state must be provided")
        use_wsl = state == 2

        if use_wsl:
            # Disable Docker mode if WSL is enabled (mutually exclusive)
            if hasattr(self, "chk_docker") and self.chk_docker.isChecked():
                self.chk_docker.setChecked(False)

            # Check if WSL is available
            try:
                result = subprocess.run(
                    ["wsl", "--list", "--quiet"],
                    capture_output=True,
                    timeout=5,
                    creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                )

                try:
                    output = result.stdout.decode("utf-16-le")
                except UnicodeError:
                    output = result.stdout.decode("utf-8", errors="ignore")

                if result.returncode != 0 or "Ubuntu" not in output:
                    raise RuntimeError("Ubuntu not found in WSL")
            except (OSError, ValueError) as e:
                QMessageBox.warning(
                    self,
                    "WSL Not Available",
                    f"WSL2 with Ubuntu is not available.\n\n"
                    f"Error: {e}\n\n"
                    "Please install WSL2 and Ubuntu:\n"
                    "  wsl --install -d Ubuntu-22.04",
                )
                self.chk_wsl.setChecked(False)
                return

            logger.info("WSL mode enabled")
            if hasattr(self, "toast_manager") and self.toast_manager:
                self.show_toast(
                    "WSL mode - full Pinocchio/Drake/Crocoddyl support", "info"
                )
        else:
            logger.info("WSL mode disabled")
            if hasattr(self, "toast_manager") and self.toast_manager:
                self.show_toast("Local Windows mode", "info")

        # Update UI status
        self.update_execution_status()

        # Update launch button text if a model is selected
        if hasattr(self, "btn_launch"):
            self.update_launch_button()

    def update_execution_status(self) -> None:
        """Update the runtime indicator label based on current selection.

        The label name uses ``Runtime:`` (not ``Mode:``) because that's
        the term the matching Settings group and the help dialog use,
        and it makes the answer to "where do my engines actually run?"
        unambiguous at a glance. WSL takes precedence over Docker if
        both are somehow checked (only one is meaningful at a time).
        """
        if not hasattr(self, "lbl_execution_mode"):
            return

        if hasattr(self, "chk_wsl") and self.chk_wsl.isChecked():
            self.lbl_execution_mode.setText("Runtime: WSL2 (Ubuntu Linux)")
            self.lbl_execution_mode.setStyleSheet(Styles.EXEC_MODE_DOCKER)
        elif hasattr(self, "chk_docker") and self.chk_docker.isChecked():
            self.lbl_execution_mode.setText("Runtime: Docker (Linux container)")
            self.lbl_execution_mode.setStyleSheet(Styles.EXEC_MODE_DOCKER)
        else:
            self.lbl_execution_mode.setText("Runtime: Native Windows")
            self.lbl_execution_mode.setStyleSheet(Styles.EXEC_MODE_WARNING)


class ThemedModalDialog(QDialog):
    """Custom themed frameless modal dialog."""

    def __init__(self, parent=None, title="Dialog", message=""):
        super().__init__(parent)
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import (
            QVBoxLayout,
            QLabel,
            QHBoxLayout,
            QPushButton,
            QGraphicsDropShadowEffect,
            QFrame,
        )

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.setProperty("class", "themed-modal")
        self.style().polish(self)

        layout = QVBoxLayout(self)

        self.frame = QFrame(self)
        self.frame.setStyleSheet(
            "QFrame { background-color: #24272e; border: 1px solid #3a3f4a; border-radius: 8px; }"
        )

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 4)
        self.frame.setGraphicsEffect(shadow)

        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(20, 20, 20, 20)
        frame_layout.setSpacing(15)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(
            "color: white; font-weight: bold; font-size: 16px; border: none; background: transparent;"
        )
        frame_layout.addWidget(lbl_title)

        lbl_msg = QLabel(message)
        lbl_msg.setStyleSheet(
            "color: #d4d4d4; font-size: 13px; border: none; background: transparent;"
        )
        lbl_msg.setWordWrap(True)
        frame_layout.addWidget(lbl_msg)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_yes = QPushButton("Yes")
        self.btn_yes.setProperty("class", "primary")
        self.btn_yes.style().polish(self.btn_yes)
        self.btn_yes.clicked.connect(self.accept)

        self.btn_no = QPushButton("No")
        self.btn_no.setProperty("class", "secondary")
        self.btn_no.style().polish(self.btn_no)
        self.btn_no.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_no)
        btn_layout.addWidget(self.btn_yes)

        frame_layout.addLayout(btn_layout)
        layout.addWidget(self.frame)
