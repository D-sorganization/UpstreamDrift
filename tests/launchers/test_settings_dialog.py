"""Tests for SettingsDialog."""

import time  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtWidgets import QWidget  # noqa: E402

from src.launchers.settings_dialog import (  # noqa: E402
    TAB_CONFIG,
    TAB_DIAGNOSTICS,
    TAB_LAYOUT,
    SettingsDialog,
    validate_tab_index,
)


def test_validate_tab_index():
    assert validate_tab_index(0) == 0
    assert validate_tab_index(1) == 1
    assert validate_tab_index(2) == 2
    with pytest.raises(ValueError):
        validate_tab_index(3)


@pytest.fixture
def parent_launcher(qapp):
    launcher = QWidget()
    launcher.btn_modify_layout = MagicMock()
    launcher.btn_modify_layout.isChecked.return_value = True

    launcher.chk_docker = MagicMock()
    launcher.chk_docker.isChecked.return_value = False

    launcher.chk_wsl = MagicMock()
    launcher.chk_wsl.isChecked.return_value = True

    launcher.chk_live = MagicMock()
    launcher.chk_live.isChecked.return_value = False

    launcher.chk_gpu = MagicMock()
    launcher.chk_gpu.isChecked.return_value = True

    launcher.available_models = {"model_1": MagicMock()}
    launcher.model_order = ["model_1"]
    launcher.model_cards = {"model_1": MagicMock()}
    launcher.selected_model = "model_1"
    launcher.docker_available = True
    launcher.registry = True

    launcher.open_layout_manager = MagicMock()
    return launcher


def test_settings_dialog_init(parent_launcher, qapp):
    data = {"summary": {"status": "healthy"}}
    dialog = SettingsDialog(parent=parent_launcher, diagnostics_data=data, initial_tab=TAB_CONFIG)

    # Check that checkboxes are synced
    assert dialog.chk_docker.isChecked() is False
    assert dialog.chk_wsl.isChecked() is True
    assert dialog.chk_live_viz.isChecked() is False
    assert dialog.chk_gpu.isChecked() is True

    assert dialog.tabs.currentIndex() == TAB_CONFIG


def test_on_reset_layout(parent_launcher, qapp):
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_LAYOUT)

    mock_slot = MagicMock()
    dialog.reset_layout_requested.connect(mock_slot)

    dialog._on_reset_layout()
    mock_slot.assert_called_once()


@patch("src.launchers.settings_dialog.DockerBuildThread")
def test_start_build(mock_thread_class, parent_launcher, qapp):
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_CONFIG)

    mock_thread = MagicMock()
    mock_thread_class.return_value = mock_thread

    dialog._start_build()

    assert dialog._btn_build.isEnabled() is False
    assert dialog._btn_cancel_build.isEnabled() is True
    assert dialog._build_status.text() == "Building..."
    mock_thread.start.assert_called_once()


def test_on_build_finished(parent_launcher, qapp):
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_CONFIG)
    dialog._build_start_time = 0
    dialog._build_timer_id = 123
    dialog.killTimer = MagicMock()

    dialog._btn_build.setEnabled(False)
    dialog._btn_cancel_build.setEnabled(True)

    dialog._on_build_finished(True, "Done")

    assert dialog._btn_build.isEnabled() is True
    assert dialog._btn_cancel_build.isEnabled() is False
    dialog.killTimer.assert_called_once_with(123)
    assert "SUCCESS" in dialog._build_status.text()


def test_cancel_build(parent_launcher, qapp):
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_CONFIG)
    dialog.build_thread = MagicMock()
    dialog.build_thread.isRunning.return_value = True
    dialog._build_timer_id = 123
    dialog.killTimer = MagicMock()

    dialog._btn_build.setEnabled(False)
    dialog._btn_cancel_build.setEnabled(True)

    dialog._cancel_build()

    dialog.build_thread.terminate.assert_called_once()
    assert dialog._btn_build.isEnabled() is True
    assert dialog._btn_cancel_build.isEnabled() is False
    assert "cancelled" in dialog._build_status.text().lower()


@patch("pathlib.Path.exists", return_value=True)
def test_load_app_log_success(mock_exists, parent_launcher, qapp):
    log_content = "Line 1\nLine 2\n"
    with patch("pathlib.Path.read_text", return_value=log_content):
        dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_DIAGNOSTICS)
        assert dialog._log_viewer.toPlainText() == "Line 1\nLine 2"


@patch("pathlib.Path.exists", return_value=False)
def test_load_app_log_fail(mock_exists, parent_launcher, qapp):
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_DIAGNOSTICS)
    assert "No log file found" in dialog._log_viewer.toPlainText()


@patch("pathlib.Path.exists", return_value=True)
def test_load_process_log_success(mock_exists, parent_launcher, qapp):
    log_content = "Process Line 1\nProcess Line 2\n"
    with patch("pathlib.Path.read_text", return_value=log_content):
        dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_DIAGNOSTICS)
        assert dialog._proc_log_viewer.toPlainText() == "Process Line 1\nProcess Line 2"


@patch("src.launchers.launcher_diagnostics.LauncherDiagnostics")
def test_refresh_diagnostics(mock_diag_class, parent_launcher, qapp):
    mock_diag = MagicMock()
    mock_diag.run_all_checks.return_value = {"summary": {"status": "degraded"}}
    mock_diag_class.return_value = mock_diag

    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_DIAGNOSTICS)

    dialog._refresh_diagnostics()

    mock_diag.run_all_checks.assert_called_once()
    assert "degraded" in dialog._diag_browser.toHtml().lower()


def test_timer_event(parent_launcher, qapp):
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_CONFIG)
    dialog._build_start_time = time.monotonic() - 5

    dialog.timerEvent(None)

    text = dialog._build_status.text()
    assert "Building..." in text
    assert "elapsed" in text


def test_settings_dialog_no_launcher():
    # Test initialization without a parent launcher to cover the 'if launcher' conditions
    dialog = SettingsDialog(parent=None, initial_tab=TAB_CONFIG)
    assert dialog.parent() is None

    # Test diagnostics refresh without a launcher parent
    with patch("src.launchers.launcher_diagnostics.LauncherDiagnostics") as mock_diag_class:
        mock_diag = MagicMock()
        # Include detailed checks, engines, and recommendations to cover render functions
        mock_diag.run_all_checks.return_value = {
            "summary": {
                "status": "degraded",
                "passed": 1,
                "failed": 0,
                "warnings": 0,
                "total_checks": 1,
            },
            "checks": [
                {
                    "name": "test_check",
                    "status": "pass",
                    "message": "OK",
                    "duration_ms": 123.4,
                },
                {
                    "name": "engine_availability",
                    "status": "warning",
                    "message": "Some engines missing",
                    "duration_ms": 10,
                    "details": {
                        "engines": [
                            {
                                "name": "drake",
                                "installed": False,
                                "missing_deps": ["pydrake"],
                                "diagnostic": "Missing pydrake",
                            },
                            {
                                "name": "mujoco",
                                "installed": True,
                                "version": "2.3.0",
                                "diagnostic": "OK",
                            },
                        ]
                    },
                },
            ],
            "recommendations": ["Do this", "Do that"],
        }
        mock_diag_class.return_value = mock_diag

        dialog._refresh_diagnostics()
        html = dialog._diag_browser.toHtml()
        assert "123ms" in html
        assert "pydrake" in html
        assert "Do this" in html


def test_load_logs_exceptions(parent_launcher, qapp):
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_DIAGNOSTICS)

    # Refresh all logs triggers both loading functions
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_text", side_effect=OSError("Boom")),
    ):
        dialog._refresh_all_logs()

    assert "No log file found" in dialog._log_viewer.toPlainText()
    assert "No process output log yet" in dialog._proc_log_viewer.toPlainText()


def test_on_build_log(parent_launcher, qapp):
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_CONFIG)

    # Should update the text and cursor
    dialog._on_build_log("Step 1/2")
    assert "Step 1/2" in dialog.build_console.toPlainText()

    # Test when scrollbar is completely fake/missing
    with patch.object(dialog.build_console, "verticalScrollBar", return_value=None):
        dialog._on_build_log("Step 2/2")


def test_on_build_finished_no_timer(parent_launcher, qapp):
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_CONFIG)
    dialog._build_start_time = 0
    # Timer not set
    if hasattr(dialog, "_build_timer_id"):
        del dialog._build_timer_id

    dialog._on_build_finished(False, "Failed early")
    assert "FAILED" in dialog._build_status.text()


def test_cancel_build_no_timer(parent_launcher, qapp):
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_CONFIG)
    dialog.build_thread = MagicMock()
    dialog.build_thread.isRunning.return_value = True

    # Timer not set
    if hasattr(dialog, "_build_timer_id"):
        del dialog._build_timer_id

    dialog._cancel_build()
    assert "cancelled" in dialog._build_status.text().lower()


def test_cancel_build_not_running(parent_launcher, qapp):
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_CONFIG)
    dialog.build_thread = MagicMock()
    dialog.build_thread.isRunning.return_value = False

    # Should not do anything
    dialog._cancel_build()
    dialog.build_thread.terminate.assert_not_called()


def test_timer_event_no_start_time(parent_launcher, qapp):
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_CONFIG)
    if hasattr(dialog, "_build_start_time"):
        del dialog._build_start_time

    # Should not raise any error
    dialog.timerEvent(None)
