"""Dialog and settings management mixin for GolfLauncher.

Contains methods for help dialogs, about dialog, shortcuts overlay,
preferences, settings, diagnostics, environment manager, layout manager,
bug reporting, and AI settings.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMessageBox,
)

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
        if dialog.exec():
            # Reload settings in panel
            if hasattr(self, "ai_panel"):
                pass

    def toggle_ai_assistant(self, checked: bool) -> None:
        """Toggle the AI Assistant panel visibility via the content splitter.

        Args:
            checked: Whether the button is checked.
        """
        if not AI_AVAILABLE or not hasattr(self, "ai_panel"):
            return

        self._ai_visible = checked
        # Keep the toggle button in sync when called programmatically
        if hasattr(self, "btn_ai") and self.btn_ai.isChecked() != checked:
            self.btn_ai.setChecked(checked)

        total = self.content_splitter.width() or 1200

        if checked:
            # Remove max-width constraint and allocate 30% to AI panel
            self.ai_panel.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
            self.content_splitter.setSizes([int(total * 0.7), int(total * 0.3)])
        else:
            # Collapse AI panel to zero width
            self.content_splitter.setSizes([total, 0])
            self.ai_panel.setMaximumWidth(0)

    def _report_bug(self) -> None:
        """Open default mail client to report a bug."""
        subject = "Bug Report: UpstreamDrift"
        body = "Please describe the issue you encountered:\n\n"

        from urllib.parse import quote

        email = "support@golfmodelingsuite.com"
        mailto_url = f"mailto:{email}?subject={quote(subject)}&body={quote(body)}"

        QDesktopServices.openUrl(QUrl(mailto_url))

    def _open_settings(self, tab: int = 0) -> None:
        """Open the settings dialog with Diagnostics and Rebuild Environment tabs.

        Args:
            tab: Initial tab index (0=Diagnostics, 1=Rebuild Environment).
        """
        diagnostics_data = None
        try:
            from src.launchers.launcher_diagnostics import LauncherDiagnostics

            diag = LauncherDiagnostics()
            diagnostics_data = diag.run_all_checks()

            diagnostics_data["runtime_state"] = {
                "available_models_count": len(self.available_models),
                "available_model_ids": list(self.available_models.keys()),
                "model_order_count": len(self.model_order),
                "model_order": self.model_order,
                "model_cards_count": len(self.model_cards),
                "selected_model": self.selected_model,
                "docker_available": self.docker_available,
                "registry_loaded": self.registry is not None,
            }
        except ImportError as e:
            logger.warning(f"Failed to run diagnostics: {e}")

        dialog = SettingsDialog(
            parent=self,
            diagnostics_data=diagnostics_data,
            initial_tab=tab,
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
        self.layout_edit_mode = checked
        self.layout_manager.set_edit_mode(checked)
        if checked:
            self.btn_modify_layout.setText("Edit Mode On")
            self.btn_modify_layout.setStyleSheet("""
                QPushButton {
                    background-color: #007acc;
                    color: white;
                    border: 1px solid #0099ff;
                }
                """)
            self.btn_customize_tiles.setEnabled(True)
            self.show_toast("Drag tiles to reorder. Double-click to launch.", "info")
        else:
            self.btn_modify_layout.setText("Layout Locked")
            self.btn_modify_layout.setStyleSheet("""
                QPushButton {
                    background-color: #444444;
                    color: #cccccc;
                }
                """)
            self.btn_customize_tiles.setEnabled(False)

    def _on_docker_mode_changed(self, state: int) -> None:
        """Handle Docker mode toggle change.

        Args:
            state: Qt checkbox state (0=unchecked, 2=checked)
        """
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

    def _on_wsl_mode_changed(self, state: int) -> None:
        """Handle WSL mode toggle change.

        Args:
            state: Qt checkbox state (0=unchecked, 2=checked)
        """
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
        """Update the execution mode label based on current settings."""
        if not hasattr(self, "lbl_execution_mode"):
            return

        if hasattr(self, "chk_wsl") and self.chk_wsl.isChecked():
            self.lbl_execution_mode.setText("Mode: WSL (Ubuntu)")
            self.lbl_execution_mode.setStyleSheet(
                "color: #30D158; font-weight: bold; margin-left: 10px;"
            )
        elif hasattr(self, "chk_docker") and self.chk_docker.isChecked():
            self.lbl_execution_mode.setText("Mode: Docker Container")
            self.lbl_execution_mode.setStyleSheet(
                "color: #30D158; font-weight: bold; margin-left: 10px;"
            )
        else:
            self.lbl_execution_mode.setText("Mode: Local (Windows)")
            self.lbl_execution_mode.setStyleSheet(
                "color: #FFD60A; font-weight: bold; margin-left: 10px;"
            )
