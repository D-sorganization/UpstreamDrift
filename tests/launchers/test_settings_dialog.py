"""Tests for SettingsDialog."""

import time
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QWidget

from src.launchers.settings_dialog import (
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
    dialog = SettingsDialog(
        parent=parent_launcher, diagnostics_data=data, initial_tab=TAB_CONFIG
    )

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
