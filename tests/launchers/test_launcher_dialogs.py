"""Tests for launcher_dialogs.py."""

from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtWidgets import QMainWindow  # noqa: E402

from src.launchers.launcher_dialogs import LauncherDialogsMixin  # noqa: E402


class DummyLauncher(QMainWindow, LauncherDialogsMixin):
    def __init__(self):
        super().__init__()
        self.available_models = {"m1": "model1"}
        self.model_order = ["m1"]
        self.model_cards = {}
        self.selected_model = None
        self.docker_available = True
        self.registry = None
        self.toast_manager = None

        # Fake attributes for tests
        self.btn_ai = MagicMock()
        self.content_splitter = MagicMock()
        self.ai_panel = MagicMock()
        self.btn_modify_layout = MagicMock()
        self.btn_customize_tiles = MagicMock()
        self.layout_manager = MagicMock()
        self.chk_wsl = MagicMock()
        self.chk_docker = MagicMock()
        self.lbl_execution_mode = MagicMock()
        self.update_launch_button = MagicMock()


@pytest.fixture
def launcher(qapp):
    return DummyLauncher()


@patch("src.launchers.launcher_dialogs.UI_COMPONENTS_AVAILABLE", True)
@patch("src.shared.python.ui.ToastManager")
def test_init_ui_components_true(mock_toast, launcher):
    with patch.object(launcher, "_setup_keyboard_shortcuts") as mock_setup:
        launcher._init_ui_components()
        mock_toast.assert_called_once_with(launcher)
        mock_setup.assert_called_once()
        assert launcher.toast_manager is not None


@patch("src.launchers.launcher_dialogs.UI_COMPONENTS_AVAILABLE", False)
def test_init_ui_components_false(launcher):
    launcher._init_ui_components()
    assert launcher.toast_manager is None


def test_setup_keyboard_shortcuts(launcher):
    with patch("src.launchers.launcher_dialogs.QShortcut") as mock_shortcut:
        launcher._setup_keyboard_shortcuts()
        assert mock_shortcut.call_count == 4


@patch("src.launchers.launcher_dialogs.HELP_SYSTEM_AVAILABLE", True)
def test_show_help_dialog_true(launcher):
    with patch("src.shared.python.gui_pkg.help_system.HelpDialog") as mock_dialog:
        instance = MagicMock()
        mock_dialog.return_value = instance
        launcher._show_help_dialog("topic")
        mock_dialog.assert_called_once_with(launcher, initial_topic="topic")
        instance.exec.assert_called_once()


@patch("src.launchers.launcher_dialogs.HELP_SYSTEM_AVAILABLE", False)
def test_show_help_dialog_false(launcher):
    with patch("src.launchers.ui_components.HelpDialog") as mock_dialog:
        instance = MagicMock()
        mock_dialog.return_value = instance
        launcher._show_help_dialog()
        mock_dialog.assert_called_once_with(launcher)
        instance.exec.assert_called_once()


@patch("src.launchers.launcher_dialogs.QDesktopServices.openUrl")
def test_open_project_map_exists(mock_open, launcher):
    with patch("src.launchers.launcher_dialogs.Path.exists", return_value=True):
        launcher._open_project_map()
        mock_open.assert_called_once()


@patch("src.launchers.launcher_dialogs.QMessageBox.warning")
def test_open_project_map_not_exists(mock_warning, launcher):
    with patch("src.launchers.launcher_dialogs.Path.exists", return_value=False):
        launcher._open_project_map()
        mock_warning.assert_called_once()


@patch("src.launchers.launcher_dialogs.QMessageBox.about")
def test_show_about_dialog(mock_about, launcher):
    launcher._show_about_dialog()
    mock_about.assert_called_once()


@patch("src.launchers.launcher_dialogs.UI_COMPONENTS_AVAILABLE", True)
def test_show_shortcuts_overlay(launcher):
    with patch("src.shared.python.ui.ShortcutsOverlay") as mock_overlay:
        instance = MagicMock()
        mock_overlay.return_value = instance
        launcher._show_shortcuts_overlay()
        instance.show.assert_called_once()
        instance.setFocus.assert_called_once()


@patch("src.launchers.launcher_dialogs.UI_COMPONENTS_AVAILABLE", False)
def test_show_shortcuts_overlay_false(launcher):
    with patch("src.shared.python.ui.ShortcutsOverlay") as mock_overlay:
        launcher._show_shortcuts_overlay()
        mock_overlay.assert_not_called()


@patch("src.launchers.launcher_dialogs.UI_COMPONENTS_AVAILABLE", True)
def test_show_preferences(launcher):
    with patch("src.shared.python.ui.PreferencesDialog") as mock_dialog:
        instance = MagicMock()
        mock_dialog.return_value = instance
        launcher._show_preferences()
        instance.exec.assert_called_once()


@patch("src.launchers.launcher_dialogs.UI_COMPONENTS_AVAILABLE", False)
def test_show_preferences_false(launcher):
    with patch("src.shared.python.ui.PreferencesDialog") as mock_dialog:
        launcher._show_preferences()
        mock_dialog.assert_not_called()


def test_show_toast(launcher):
    launcher.toast_manager = MagicMock()
    launcher.show_toast("msg", "success")
    launcher.toast_manager.show_success.assert_called_with("msg")

    launcher.show_toast("msg", "error")
    launcher.toast_manager.show_error.assert_called_with("msg")

    launcher.show_toast("msg", "warning")
    launcher.toast_manager.show_warning.assert_called_with("msg")

    launcher.show_toast("msg", "info")
    launcher.toast_manager.show_info.assert_called_with("msg")


@patch("src.launchers.launcher_dialogs.AI_AVAILABLE", True)
def test_open_ai_settings(launcher):
    with patch("src.shared.python.ai.gui.AISettingsDialog") as mock_dialog:
        instance = MagicMock()
        instance.exec.return_value = True
        mock_dialog.return_value = instance
        launcher._open_ai_settings()
        mock_dialog.assert_called_with(launcher)


@patch("src.launchers.launcher_dialogs.AI_AVAILABLE", False)
def test_open_ai_settings_not_available(launcher):
    with patch("src.shared.python.ai.gui.AISettingsDialog") as mock_dialog:
        launcher._open_ai_settings()
        mock_dialog.assert_not_called()


@patch("src.launchers.launcher_dialogs.AI_AVAILABLE", False)
def test_toggle_ai_assistant_not_available(launcher):
    launcher.toggle_ai_assistant(True)
    launcher.content_splitter.setSizes.assert_not_called()


@patch("src.launchers.launcher_dialogs.AI_AVAILABLE", True)
def test_toggle_ai_assistant(launcher):
    launcher.content_splitter.width.return_value = 1000
    launcher.btn_ai.isChecked.return_value = False

    launcher.toggle_ai_assistant(True)
    launcher.btn_ai.setChecked.assert_called_with(True)
    launcher.content_splitter.setSizes.assert_called_with([700, 300])

    launcher.toggle_ai_assistant(False)
    launcher.content_splitter.setSizes.assert_called_with([1000, 0])


@patch("src.launchers.launcher_dialogs.QDesktopServices.openUrl")
def test_report_bug(mock_open, launcher):
    launcher._report_bug()
    mock_open.assert_called_once()


@patch("src.launchers.launcher_dialogs.SettingsDialog")
def test_open_settings(mock_dialog, launcher):
    instance = MagicMock()
    mock_dialog.return_value = instance
    with patch("src.launchers.launcher_diagnostics.LauncherDiagnostics") as mock_diag:
        diag_inst = MagicMock()
        diag_inst.run_all_checks.return_value = {}
        mock_diag.return_value = diag_inst

        launcher._open_settings(tab=1)
        mock_dialog.assert_called_once()
        instance.exec.assert_called_once()


def test_open_diagnostics(launcher):
    with patch.object(launcher, "_open_settings") as mock_open:
        launcher.open_diagnostics()
        mock_open.assert_called_with(tab=2)


def test_open_environment_manager(launcher):
    with patch.object(launcher, "_open_settings") as mock_open:
        launcher.open_environment_manager()
        mock_open.assert_called_with(tab=1)


def test_reset_layout_to_defaults(launcher):
    with (
        patch("src.launchers.launcher_dialogs.Path.exists", return_value=True),
        patch("src.launchers.launcher_dialogs.Path.with_suffix"),
        patch("src.launchers.launcher_dialogs.Path.rename") as mock_rename,
        patch(
            "src.launchers.launcher_dialogs.Path.home",
            return_value=Path("C:/Users/test"),
        ),
    ):
        launcher._initialize_model_order = MagicMock()
        launcher._sync_model_cards = MagicMock()
        launcher._rebuild_grid = MagicMock()

        launcher._reset_layout_to_defaults()

        mock_rename.assert_called_once()
        launcher._initialize_model_order.assert_called_once()


def test_reset_layout_to_defaults_not_exists(launcher):
    with (
        patch("src.launchers.launcher_dialogs.Path.exists", return_value=False),
        patch(
            "src.launchers.launcher_dialogs.Path.home",
            return_value=Path("C:/Users/test"),
        ),
    ):
        launcher._initialize_model_order = MagicMock()
        launcher._sync_model_cards = MagicMock()
        launcher._rebuild_grid = MagicMock()

        launcher._reset_layout_to_defaults()
        launcher._initialize_model_order.assert_called_once()


def test_reset_layout_to_defaults_error(launcher):
    with (
        patch("src.launchers.launcher_dialogs.Path.exists", side_effect=OSError("err")),
        patch.object(launcher, "show_toast") as mock_toast,
    ):
        launcher._reset_layout_to_defaults()
        mock_toast.assert_called_once()


def test_open_help(launcher):
    with patch.object(launcher, "_show_help_dialog") as mock_show:
        launcher.open_help()
        mock_show.assert_called_once()


@patch("src.launchers.launcher_dialogs.LayoutManagerDialog")
def test_open_layout_manager(mock_dialog, launcher):
    instance = MagicMock()
    instance.exec.return_value = True
    instance.selected_ids.return_value = ["m1"]
    mock_dialog.return_value = instance

    launcher._apply_model_selection = MagicMock()
    launcher.open_layout_manager()
    launcher._apply_model_selection.assert_called_with(["m1"])

    instance.exec.return_value = False
    launcher.open_layout_manager()
    assert launcher._apply_model_selection.call_count == 1


def test_toggle_layout_mode(launcher):
    launcher.toggle_layout_mode(True)
    assert launcher.layout_edit_mode is True
    launcher.btn_customize_tiles.setEnabled.assert_called_with(True)

    launcher.toggle_layout_mode(False)
    launcher.btn_customize_tiles.setEnabled.assert_called_with(False)


def test_on_docker_mode_changed(launcher):
    launcher.chk_wsl.isChecked.return_value = True
    launcher.docker_available = True
    launcher.toast_manager = MagicMock()

    with patch.object(launcher, "update_execution_status") as mock_exec:
        launcher.btn_launch = MagicMock()
        launcher._on_docker_mode_changed(2)
        launcher.chk_wsl.setChecked.assert_called_with(False)
        mock_exec.assert_called_once()
        launcher.update_launch_button.assert_called_once()
        launcher.toast_manager.show_info.assert_called()


def test_on_docker_mode_changed_disable(launcher):
    launcher.toast_manager = MagicMock()
    with patch.object(launcher, "update_execution_status"):
        launcher._on_docker_mode_changed(0)
        launcher.toast_manager.show_info.assert_called_with(
            "Local mode - engines will run on host system"
        )


@patch("src.launchers.launcher_dialogs.QMessageBox.warning")
def test_on_docker_mode_changed_unavailable(mock_warning, launcher):
    launcher.docker_available = False
    launcher._on_docker_mode_changed(2)
    mock_warning.assert_called_once()


@patch("src.launchers.launcher_dialogs.subprocess.run")
def test_on_wsl_mode_changed(mock_run, launcher):
    launcher.chk_docker.isChecked.return_value = True
    launcher.toast_manager = MagicMock()

    mock_result = MagicMock()
    mock_result.stdout = "Ubuntu-22.04".encode("utf-16-le")
    mock_result.returncode = 0
    mock_run.return_value = mock_result

    with patch.object(launcher, "update_execution_status") as mock_exec:
        launcher.btn_launch = MagicMock()  # trigger update_launch_button branch
        launcher._on_wsl_mode_changed(2)
        launcher.chk_docker.setChecked.assert_called_with(False)
        mock_exec.assert_called_once()
        launcher.update_launch_button.assert_called_once()
        launcher.toast_manager.show_info.assert_called()


@patch("src.launchers.launcher_dialogs.subprocess.run")
def test_on_wsl_mode_changed_utf8_fallback(mock_run, launcher):
    launcher.chk_docker.isChecked.return_value = True

    mock_result = MagicMock()
    mock_stdout = MagicMock()
    mock_stdout.decode.side_effect = [
        UnicodeDecodeError("utf-16", b"", 0, 1, "err"),
        "Ubuntu",
    ]
    mock_result.stdout = mock_stdout
    mock_result.returncode = 0
    mock_run.return_value = mock_result

    with patch.object(launcher, "update_execution_status"):
        launcher._on_wsl_mode_changed(2)
    launcher.chk_docker.setChecked.assert_called_with(False)


@patch("src.launchers.launcher_dialogs.subprocess.run")
@patch("src.launchers.launcher_dialogs.QMessageBox.warning")
def test_on_wsl_mode_changed_error(mock_warning, mock_run, launcher):
    mock_run.side_effect = OSError("err")
    launcher._on_wsl_mode_changed(2)
    mock_warning.assert_called_once()


def test_on_wsl_mode_changed_disable(launcher):
    launcher.toast_manager = MagicMock()
    with patch.object(launcher, "update_execution_status"):
        launcher._on_wsl_mode_changed(0)
        launcher.toast_manager.show_info.assert_called_with("Local Windows mode")


def test_update_execution_status(launcher):
    launcher.chk_wsl.isChecked.return_value = True
    launcher.update_execution_status()
    launcher.lbl_execution_mode.setText.assert_called_with("Mode: WSL (Ubuntu)")

    launcher.chk_wsl.isChecked.return_value = False
    launcher.chk_docker.isChecked.return_value = True
    launcher.update_execution_status()
    launcher.lbl_execution_mode.setText.assert_called_with("Mode: Docker Container")

    launcher.chk_docker.isChecked.return_value = False
    launcher.update_execution_status()
    launcher.lbl_execution_mode.setText.assert_called_with("Mode: Local (Windows)")


def test_update_execution_status_no_label(launcher):
    del launcher.lbl_execution_mode
    launcher.update_execution_status()  # Should simply return
