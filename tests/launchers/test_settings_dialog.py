"""Tests for SettingsWidget."""

# mypy: disable-error-code="attr-defined,method-assign"

import time  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtWidgets import QWidget  # noqa: E402
from src.launchers.settings_dialog import (  # noqa: E402
    TAB_APPEARANCE,
    TAB_CONFIG,
    TAB_DIAGNOSTICS,
    TAB_LAYOUT,
    TAB_NOTIFICATIONS,
    TAB_PERFORMANCE,
    TAB_STARTUP,
    SettingsDialog,
    SettingsWidget,
    validate_tab_index,
)


def test_validate_tab_index() -> None:
    """Validate all legal tab indexes; ensure out-of-range raises ValueError."""
    assert validate_tab_index(0) == 0  # TAB_LAYOUT
    assert validate_tab_index(1) == 1  # TAB_CONFIG
    assert validate_tab_index(2) == 2  # TAB_DIAGNOSTICS
    # TAB_MCP_SERVERS = 3 is a valid index (added in UpstreamDrift #5688)
    assert validate_tab_index(3) == 3  # TAB_MCP_SERVERS
    assert validate_tab_index(TAB_APPEARANCE) == TAB_APPEARANCE
    assert validate_tab_index(TAB_STARTUP) == TAB_STARTUP
    assert validate_tab_index(TAB_NOTIFICATIONS) == TAB_NOTIFICATIONS
    assert validate_tab_index(TAB_PERFORMANCE) == TAB_PERFORMANCE
    with pytest.raises(ValueError):
        validate_tab_index(8)  # out of range


@pytest.fixture
def parent_launcher(qapp) -> QWidget:
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


def test_settings_dialog_init(parent_launcher, qapp) -> None:
    data = {"summary": {"status": "healthy"}}
    dialog = SettingsWidget(
        parent=parent_launcher, diagnostics_data=data, initial_tab=TAB_CONFIG
    )

    # Check that checkboxes are synced
    assert dialog.chk_docker.isChecked() is False
    assert dialog.chk_wsl.isChecked() is True
    assert dialog.chk_live_viz.isChecked() is False
    assert dialog.chk_gpu.isChecked() is True

    assert dialog.tabs.currentIndex() == TAB_CONFIG


def test_settings_widget_accepts_preferences_tab(parent_launcher, qapp) -> None:
    """Preferences shortcut uses tab 4; settings startup must accept it."""
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_APPEARANCE)

    assert dialog.tabs.currentIndex() == TAB_APPEARANCE


def test_settings_dialog_wrapper_keeps_legacy_dialog_contract(
    parent_launcher, qapp
) -> None:
    """Legacy imports still get a QDialog-like settings surface."""
    dialog = SettingsDialog(parent=parent_launcher, initial_tab=TAB_CONFIG)

    assert dialog.tabs.currentIndex() == TAB_CONFIG
    assert dialog.widget.parent() is dialog


def test_on_reset_layout(parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_LAYOUT)

    mock_slot = MagicMock()
    dialog.reset_layout_requested.connect(mock_slot)

    dialog._on_reset_layout()
    mock_slot.assert_called_once()


@patch("src.launchers.settings_dialog.DockerBuildThread")
def test_start_build(mock_thread_class, parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)

    mock_thread = MagicMock()
    mock_thread_class.return_value = mock_thread

    dialog._start_build()

    assert dialog._btn_build.isEnabled() is False
    assert dialog._btn_cancel_build.isEnabled() is True
    assert dialog._build_status.text() == "Building..."
    mock_thread.start.assert_called_once()


def test_on_build_finished(parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)
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


def test_cancel_build(parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)
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
def test_load_app_log_success(mock_exists, parent_launcher, qapp) -> None:
    log_content = "Line 1\nLine 2\n"
    with patch("pathlib.Path.read_text", return_value=log_content):
        dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_DIAGNOSTICS)
        assert dialog._log_viewer.toPlainText() == "Line 1\nLine 2"


@patch("pathlib.Path.exists", return_value=False)
def test_load_app_log_fail(mock_exists, parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_DIAGNOSTICS)
    assert "No log file found" in dialog._log_viewer.toPlainText()


@patch("pathlib.Path.exists", return_value=True)
def test_load_process_log_success(mock_exists, parent_launcher, qapp) -> None:
    log_content = "Process Line 1\nProcess Line 2\n"
    with patch("pathlib.Path.read_text", return_value=log_content):
        dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_DIAGNOSTICS)
        assert dialog._proc_log_viewer.toPlainText() == "Process Line 1\nProcess Line 2"


@patch("src.launchers.launcher_diagnostics.LauncherDiagnostics")
def test_refresh_diagnostics(mock_diag_class, parent_launcher, qapp) -> None:
    mock_diag = MagicMock()
    mock_diag.run_all_checks.return_value = {"summary": {"status": "degraded"}}
    mock_diag_class.return_value = mock_diag

    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_DIAGNOSTICS)

    dialog._refresh_diagnostics()

    mock_diag.run_all_checks.assert_called_once()
    assert "degraded" in dialog._diag_browser.toHtml().lower()


def test_timer_event(parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)
    dialog._build_start_time = time.monotonic() - 5

    dialog.timerEvent(None)

    text = dialog._build_status.text()
    assert "Building..." in text
    assert "elapsed" in text


def test_settings_dialog_no_launcher() -> None:
    # Test initialization without a parent launcher to cover the 'if launcher' conditions
    dialog = SettingsWidget(parent=None, initial_tab=TAB_CONFIG)
    assert dialog.parent() is None

    # Test diagnostics refresh without a launcher parent
    with patch(
        "src.launchers.launcher_diagnostics.LauncherDiagnostics"
    ) as mock_diag_class:
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


def test_load_logs_exceptions(parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_DIAGNOSTICS)

    # Refresh all logs triggers both loading functions
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_text", side_effect=OSError("Boom")),
    ):
        dialog._refresh_all_logs()

    assert "No log file found" in dialog._log_viewer.toPlainText()
    assert "No process output log yet" in dialog._proc_log_viewer.toPlainText()


def test_on_build_log(parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)

    # Should update the text and cursor
    dialog._on_build_log("Step 1/2")
    assert "Step 1/2" in dialog.build_console.toPlainText()

    # Test when scrollbar is completely fake/missing
    with patch.object(dialog.build_console, "verticalScrollBar", return_value=None):
        dialog._on_build_log("Step 2/2")


def test_on_build_finished_no_timer(parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)
    dialog._build_start_time = 0
    # Timer not set
    if hasattr(dialog, "_build_timer_id"):
        del dialog._build_timer_id

    dialog._on_build_finished(False, "Failed early")
    assert "FAILED" in dialog._build_status.text()


def test_cancel_build_no_timer(parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)
    dialog.build_thread = MagicMock()
    dialog.build_thread.isRunning.return_value = True

    # Timer not set
    if hasattr(dialog, "_build_timer_id"):
        del dialog._build_timer_id

    dialog._cancel_build()
    assert "cancelled" in dialog._build_status.text().lower()


def test_cancel_build_not_running(parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)
    dialog.build_thread = MagicMock()
    dialog.build_thread.isRunning.return_value = False

    # Should not do anything
    dialog._cancel_build()
    dialog.build_thread.terminate.assert_not_called()


def test_timer_event_no_start_time(parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)
    if hasattr(dialog, "_build_start_time"):
        del dialog._build_start_time

    # Should not raise any error
    dialog.timerEvent(None)


# ── Issue #5730 / #5723: ViewMode import and zoom slider wiring ──────────────


def test_layout_tab_view_mode_combo_populated(parent_launcher, qapp) -> None:
    """#5730: ViewMode combo must be populated (import from correct module)."""
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_LAYOUT)
    # If the import is broken the combo stays empty (count == 0)
    assert dialog.combo_view_mode.count() > 0, (
        "combo_view_mode has no items — ViewMode import failed (check #5730)"
    )


def test_layout_tab_zoom_slider_calls_on_zoom_slider_changed(
    parent_launcher, qapp
) -> None:
    """#5723: Moving the zoom slider must call _on_zoom_slider_changed, not _zoom_slider_changed."""
    from src.launchers.launcher_constants import ViewMode
    from src.launchers.launcher_layout_manager import LayoutManager

    layout_manager = MagicMock(spec=LayoutManager)
    layout_manager.view_mode = ViewMode.LIST_LARGE
    layout_manager.tile_scale = 1.0
    parent_launcher.layout_manager = layout_manager
    parent_launcher._scale_to_slider = MagicMock(return_value=50)
    parent_launcher._slider_to_scale = MagicMock(return_value=0.75)
    # The real method name is _on_zoom_slider_changed (not _zoom_slider_changed)
    parent_launcher._on_zoom_slider_changed = MagicMock()
    # Ensure the wrong name is not present (so hasattr check routes correctly)
    if hasattr(parent_launcher, "_zoom_slider_changed"):
        del parent_launcher._zoom_slider_changed

    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_LAYOUT)

    # Simulate slider movement
    dialog.zoom_slider.setValue(75)

    parent_launcher._on_zoom_slider_changed.assert_called_with(75)


def test_configuration_tab_uses_scroll_area(parent_launcher, qapp) -> None:
    """TDD test: Configuration tab must return a QScrollArea wrapping the container widget."""
    from PyQt6.QtWidgets import QScrollArea

    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)
    config_tab = dialog.tabs.widget(TAB_CONFIG)

    assert isinstance(config_tab, QScrollArea), (
        "Configuration tab must be a QScrollArea"
    )
    assert config_tab.widget() is not None, "QScrollArea must have a wrapped widget"
    assert config_tab.widgetResizable(), "QScrollArea widgetResizable must be True"


def test_compare_versions(parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)
    assert dialog._compare_versions("1.26.4", ">=1.26.4") is True
    assert dialog._compare_versions("1.27.0", ">=1.26.4") is True
    assert dialog._compare_versions("1.25.0", ">=1.26.4") is False
    assert dialog._compare_versions("Unknown", ">=1.26.4") is True
    assert dialog._compare_versions("Missing", ">=1.26.4") is False


@patch("PyQt6.QtWidgets.QMessageBox.information")
def test_check_windows_deps(mock_info, parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)
    dialog._check_windows_deps()
    mock_info.assert_called_once()
    args, kwargs = mock_info.call_args
    assert "Windows Environment" in args[2]


@patch("PyQt6.QtWidgets.QMessageBox.warning")
@patch("src.launchers.docker_manager.get_docker_cmd", return_value=["docker"])
@patch("subprocess.run")
def test_check_docker_deps_missing_image(
    mock_run, mock_cmd, mock_warning, parent_launcher, qapp
) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)

    # Mock subprocess.run for docker image inspect to fail (returncode=1)
    mock_res = MagicMock()
    mock_res.returncode = 1
    mock_run.return_value = mock_res

    dialog._check_docker_deps()
    mock_warning.assert_called_once()
    args, kwargs = mock_warning.call_args
    assert "Missing Image" in args[2]


@patch("PyQt6.QtWidgets.QMessageBox.information")
@patch("src.launchers.docker_manager.get_docker_cmd", return_value=["docker"])
@patch("subprocess.run")
def test_check_docker_deps_success(
    mock_run, mock_cmd, mock_info, parent_launcher, qapp
) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)

    # First call: docker image inspect (returncode=0)
    # Second call: docker run (returncode=0)
    res_inspect = MagicMock()
    res_inspect.returncode = 0

    res_run = MagicMock()
    res_run.returncode = 0
    res_run.stdout = "numpy:1.26.4,scipy:1.13.1,mujoco:3.6.0,pydrake:1.22.0,pinocchio:2.6.0,opensim:4.4.0\n"

    mock_run.side_effect = [res_inspect, res_run]

    dialog._check_docker_deps()
    mock_info.assert_called_once()
    args, kwargs = mock_info.call_args
    assert "Docker Container" in args[2]


@patch("PyQt6.QtWidgets.QMessageBox.information")
@patch("subprocess.run")
@patch("pathlib.Path.exists", autospec=True)
def test_check_wsl_deps_success(
    mock_exists, mock_run, mock_info, parent_launcher, qapp
) -> None:
    mock_exists.side_effect = lambda *args, **kwargs: any(
        ".venv-wsl" in str(arg) for arg in args
    )
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)

    res_run = MagicMock()
    res_run.returncode = 0
    res_run.stdout = "numpy:1.26.4,scipy:1.13.1,mujoco:3.6.0,pydrake:1.22.0,pinocchio:2.6.0,opensim:4.4.0\n"
    mock_run.return_value = res_run

    dialog._check_wsl_deps()
    mock_info.assert_called_once()
    args, kwargs = mock_info.call_args
    assert "WSL2 Environment Status" in args[2]


@patch("PyQt6.QtWidgets.QDialog.exec")
def test_show_wsl_setup_dialog(mock_exec, parent_launcher, qapp) -> None:
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)
    dialog._show_wsl_setup_dialog()
    mock_exec.assert_called_once()


def test_launcher_ref_used_when_parent_is_none(qapp) -> None:
    """Issue #6508: when launcher kwarg is passed without parent, self._launcher
    must be used — not self.parent() — so layout and diagnostics tabs work."""
    launcher = QWidget()
    launcher.btn_modify_layout = MagicMock()
    launcher.btn_modify_layout.isChecked.return_value = True
    launcher.chk_docker = MagicMock()
    launcher.chk_docker.isChecked.return_value = True
    launcher.chk_wsl = MagicMock()
    launcher.chk_wsl.isChecked.return_value = False
    launcher.chk_live = MagicMock()
    launcher.chk_live.isChecked.return_value = True
    launcher.chk_gpu = MagicMock()
    launcher.chk_gpu.isChecked.return_value = False
    launcher.available_models = {}
    launcher.model_order = []
    launcher.model_cards = {}
    launcher.selected_model = None
    launcher.docker_available = False
    launcher.registry = None
    launcher.open_layout_manager = MagicMock()

    # Production path: launcher passed explicitly, no parent
    dialog = SettingsWidget(launcher=launcher, initial_tab=TAB_LAYOUT)

    assert dialog.parent() is None, "Qt parent should be None in this path"
    assert dialog._launcher is launcher, "_launcher must reference the passed launcher"
    # Layout tab sync: btn_modify_layout.isChecked was called during _setup_layout_tab
    launcher.btn_modify_layout.isChecked.assert_called()


def test_settings_dialog_tier_details(parent_launcher, qapp) -> None:
    """Verify that tier_details browser is instantiated and updates when stage combo changes."""
    dialog = SettingsWidget(parent=parent_launcher, initial_tab=TAB_CONFIG)
    assert hasattr(dialog, "tier_details")
    assert dialog.tier_details is not None

    # Change stage combo text
    dialog.combo_stage.setCurrentText("all")
    # Verify tier details text gets updated
    html = dialog.tier_details.toHtml()
    assert html != ""
