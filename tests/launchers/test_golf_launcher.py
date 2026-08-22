"""Tests for upstream_drift_launcher.py."""

import contextlib  # noqa: E402
from collections.abc import Generator  # noqa: E402
from types import SimpleNamespace
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtWidgets import QDialog  # noqa: E402
from src.launchers.launcher_sidekick_sidebar import SidekickSidebarManager  # noqa: E402
from src.launchers.upstream_drift_launcher import (  # noqa: E402
    UpstreamDriftLauncher,
    main,
)
from src.launchers.ui_components import StartupResults  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def startup_results() -> StartupResults:
    results = StartupResults()
    results.startup_time_ms = 100
    results.docker_available = True
    results.registry = MagicMock()
    results.engine_manager = MagicMock()
    return results


@contextlib.contextmanager
def patch_launcher_ui() -> Generator[None, None, None]:
    with (
        patch("src.launchers.upstream_drift_launcher.DockerCheckThread"),
        patch("src.launchers.upstream_drift_launcher.QTimer"),
    ):
        yield


def test_onboarding_is_skipped_in_non_interactive_runs(monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAMDRIFT_DISABLE_ONBOARDING", "1")

    with patch(
        "src.launchers.onboarding_dialog.show_onboarding_if_needed"
    ) as mock_show:
        launcher = SimpleNamespace()
        launcher.sidekick_sidebar_manager = SidekickSidebarManager(launcher)
        UpstreamDriftLauncher._show_onboarding_if_needed(launcher)

    mock_show.assert_not_called()


def test_init_managers_installs_sidekick_extensions_before_tool_bootstrap() -> None:
    launcher = MagicMock()
    order: list[str] = []
    launcher.ui_setup_manager._setup_process_console.return_value = None
    launcher.ui_setup_manager._on_process_output = MagicMock()
    launcher.ui_setup_manager.update_running_processes_ui = MagicMock()
    launcher.sidekick_sidebar_manager._install_sidekick_import_paths.side_effect = (
        lambda: order.append("extensions")
    )

    with (
        patch("src.launchers.upstream_drift_launcher.ProcessManager") as process_cls,
        patch("src.launchers.upstream_drift_launcher.ModelHandlerRegistry"),
        patch("src.launchers.upstream_drift_launcher.DockerLauncher"),
        patch(
            "src.launchers.upstream_drift_launcher.bootstrap_embeddable_tools",
            side_effect=lambda: order.append("bootstrap"),
        ),
    ):
        process_cls.return_value.running_processes = {}
        UpstreamDriftLauncher._init_managers(launcher)

    assert order[:2] == ["extensions", "bootstrap"]


def test_init_without_results(qapp) -> None:
    with (
        patch_launcher_ui(),
        patch(
            "src.launchers.launcher_orchestrator._lazy_load_model_registry"
        ) as mock_reg,
        patch(
            "src.launchers.launcher_orchestrator._lazy_load_engine_manager"
        ) as mock_eng,
    ):
        mock_reg.return_value = MagicMock()
        mock_eng.return_value = (MagicMock(), MagicMock())
        launcher = UpstreamDriftLauncher()
        assert launcher._startup_time_ms == 0
        assert launcher.docker_available is False
        mock_reg.assert_called_once()
        mock_eng.assert_called_once()


def test_init_with_results(qapp, startup_results) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher(startup_results)
        assert launcher._startup_time_ms == 100
        assert launcher.docker_available is True
        assert launcher.registry == startup_results.registry
        assert launcher.engine_manager == startup_results.engine_manager


def test_load_window_icon(qapp) -> None:
    with patch_launcher_ui():
        # Do not mock QIcon. Just instantiate and test it doesn't crash.
        launcher = UpstreamDriftLauncher()
        assert launcher.windowIcon() is not None


def test_window_icon_declares_app_user_model_id(qapp) -> None:
    """Regression: the launcher must declare the Windows AppUserModelID so the
    taskbar shows the app icon instead of the python.exe icon. Earlier fixes
    only set the window icon and asserted it was non-None, which stayed true
    even while the taskbar icon was wrong — so this asserts the identity call
    actually happens, with the correct id."""
    from src.launchers.upstream_drift_launcher import _APP_USER_MODEL_ID

    with (
        patch_launcher_ui(),
        patch("src.launchers.upstream_drift_launcher.apply_window_icon") as mock_apply,
    ):
        UpstreamDriftLauncher()

    mock_apply.assert_called_once()
    assert mock_apply.call_args.kwargs.get("app_id") == _APP_USER_MODEL_ID


def test_init_registry_exception(qapp) -> None:
    with patch(
        "src.launchers.launcher_orchestrator._lazy_load_model_registry",
        side_effect=ImportError("test"),
    ):
        launcher = UpstreamDriftLauncher()
        assert launcher.registry is None


def test_init_engine_manager_exception(qapp) -> None:
    with patch(
        "src.launchers.launcher_orchestrator._lazy_load_engine_manager",
        side_effect=RuntimeError("test"),
    ):
        launcher = UpstreamDriftLauncher()
        assert launcher.engine_manager is None


def test_build_available_models(qapp, startup_results) -> None:
    model1 = MagicMock(id="m1", type="sim")
    model2 = MagicMock(id="m2", type="utility")
    startup_results.registry.get_all_models.return_value = [model1, model2]

    launcher = UpstreamDriftLauncher(startup_results)
    assert "m1" in launcher.available_models
    assert "m2" in launcher.special_app_lookup


def test_get_model(qapp) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        launcher.available_models["m1"] = "Model1"
        assert launcher._get_model("m1") == "Model1"

        launcher.registry = MagicMock()
        launcher.registry.get_model.return_value = "ModelX"
        assert launcher._get_model("mx") == "ModelX"

        launcher.registry = None
        assert launcher._get_model("mx") is None


def test_layout_management(qapp) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        launcher.layout_manager = MagicMock()

        launcher._save_layout()
        launcher.layout_manager.save_layout.assert_called_once()

        launcher._sync_model_cards()
        launcher.layout_manager.sync_model_cards.assert_called_once()

        launcher.grid_layout = MagicMock()
        launcher._rebuild_grid()
        launcher.layout_manager.rebuild_grid.assert_called_once()

        launcher.update_launch_button = MagicMock()
        launcher._apply_model_selection(["m1", "m2"])
        launcher.layout_manager.apply_model_selection.assert_called_with(["m1", "m2"])

        launcher._swap_models("m1", "m2")
        launcher.layout_manager.swap_models.assert_called_with("m1", "m2")

        launcher.update_search_filter("test")
        launcher.layout_manager.update_search_filter.assert_called_with("test")


def test_launch_model_direct(qapp) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        launcher.select_model = MagicMock()
        launcher.launch_simulation = MagicMock()
        launcher.launch_model_direct("m1")
        launcher.select_model.assert_called_with("m1")
        launcher.launch_simulation.assert_called_once()


def test_center_window(qapp) -> None:
    from PyQt6.QtCore import QRect

    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        mock_screen = MagicMock()
        mock_screen.availableGeometry.return_value = QRect(0, 0, 1920, 1080)
        launcher.setGeometry = MagicMock()

        with patch(
            "src.launchers.upstream_drift_launcher.QApplication.primaryScreen",
            return_value=mock_screen,
        ):
            launcher.center_window()

        launcher.setGeometry.assert_called_once()


def test_load_layout_empty(qapp) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        launcher.layout_manager = MagicMock()
        launcher.layout_manager.load_layout.return_value = None
        launcher._rebuild_grid = MagicMock()
        launcher._load_layout()
        launcher._rebuild_grid.assert_called_once()


def test_load_layout_with_data(qapp) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        launcher.layout_manager = MagicMock()
        layout_data = {
            "window_geometry": {"x": 10, "y": 10, "width": 800, "height": 600},
            "options": {
                "live_visualization": False,
                "gpu_acceleration": True,
                "docker_mode": True,
            },
            "selected_model": "m1",
        }
        launcher.layout_manager.load_layout.return_value = layout_data
        launcher.docker_available = True
        launcher.model_cards = {"m1": MagicMock()}
        launcher.select_model = MagicMock()

        launcher._load_layout()
        assert not launcher.chk_live.isChecked()
        assert launcher.chk_gpu.isChecked()
        assert launcher.chk_docker.isChecked()
        launcher.select_model.assert_called_with("m1")


def test_select_model(qapp) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        card_m1 = MagicMock()
        card_m2 = MagicMock()
        launcher.model_cards = {"m1": card_m1, "m2": card_m2}

        # Mock _get_model so update_launch_button is actually called
        mock_model = MagicMock()
        mock_model.name = "Test Model"
        launcher._get_model = MagicMock(return_value=mock_model)

        launcher.update_launch_button = MagicMock()
        launcher.context_help = MagicMock()

        launcher.select_model("m1")
        assert launcher.selected_model == "m1"
        card_m1.set_selected.assert_called_once_with(True)
        card_m2.set_selected.assert_called_once_with(False)
        launcher.update_launch_button.assert_called_once_with("Test Model")
        launcher.context_help.update_context.assert_called_with("m1")


def test_update_launch_button(qapp) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()

        # None selected
        launcher.selected_model = None
        launcher.update_launch_button()
        assert not launcher.btn_launch.isEnabled()
        assert launcher.btn_launch.text() == "Select a Model"

        # Selected, docker required, unavailable
        model = MagicMock(requires_docker=True)
        launcher._get_model = MagicMock(return_value=model)
        launcher.selected_model = "m1"
        launcher.docker_available = False
        launcher.update_launch_button("M1")
        assert not launcher.btn_launch.isEnabled()
        assert "! Docker Required" in launcher.btn_launch.text()

        # Selected, ready
        model = MagicMock(requires_docker=False)
        launcher._get_model = MagicMock(return_value=model)
        launcher.selected_model = "m1"
        launcher.update_launch_button("M1")
        assert launcher.btn_launch.isEnabled()
        assert "Launch M1 >" in launcher.btn_launch.text()


def test_update_launch_button_tolerates_theme_without_success_hover() -> None:
    launcher = SimpleNamespace(
        selected_model="sidekick",
        btn_launch=MagicMock(),
        orchestrator=SimpleNamespace(docker_available=True),
        _get_model=MagicMock(return_value=SimpleNamespace(requires_docker=False)),
    )
    colors = SimpleNamespace(success="#30d158")

    with patch(
        "src.shared.python.theme.palette.get_current_colors",
        return_value=colors,
    ):
        UpstreamDriftLauncher.update_launch_button(launcher, "Sidekick")

    stylesheet = launcher.btn_launch.setStyleSheet.call_args.args[0]
    assert "background-color: #30d158;" in stylesheet


@patch("src.launchers.upstream_drift_launcher._lazy_load_engine_manager")
def test_get_engine_type(mock_lazy_em, qapp) -> None:
    with patch_launcher_ui():
        EngineType = MagicMock()
        EngineType.MUJOCO = "mujoco_enum"
        EngineType.DRAKE = "drake_enum"
        EngineType.PINOCCHIO = "pinocchio_enum"
        EngineType.OPENSIM = "opensim_enum"
        EngineType.MYOSIM = "myosim_enum"
        # EM must be callable.
        mock_em = MagicMock()
        mock_lazy_em.return_value = (mock_em, EngineType)

        launcher = UpstreamDriftLauncher()
        assert launcher._get_engine_type("mujoco") == "mujoco_enum"
        assert launcher._get_engine_type("drake") == "drake_enum"
        assert launcher._get_engine_type("pinocchio") == "pinocchio_enum"
        assert launcher._get_engine_type("opensim") == "opensim_enum"
        assert launcher._get_engine_type("myosim") == "myosim_enum"
        assert launcher._get_engine_type("unknown") == "mujoco_enum"


def test_docker_status(qapp) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        launcher.update_launch_button = MagicMock()

        launcher._apply_docker_status(True)
        assert launcher.lbl_status.text() == "System Ready"

        launcher._apply_docker_status(False)
        assert launcher.lbl_status.text() == "Docker Not Found"


def test_check_docker(qapp) -> None:
    # patch_launcher_ui already mocks DockerCheckThread, so we don't need a separate patch block.
    # However we can just grab it off the launcher to assert on it.
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        # _init_ automatically calls check_docker and starts it.
        # It was mocked globally as DockerCheckThread, let's reset it and test check_docker manually.
        launcher.docker_checker.reset_mock()

        # Second time, already running
        launcher.docker_checker.isRunning.return_value = True
        launcher.check_docker()
        launcher.docker_checker.wait.assert_called()


def test_menu_toggles(qapp) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        launcher.toggle_layout_mode = MagicMock()
        launcher.context_help = MagicMock()

        # #8023: the menu handler must not touch the phantom
        # ``btn_modify_layout`` attribute — dereferencing it raised
        # AttributeError inside a Qt slot, which aborts the process.
        launcher._toggle_layout_mode_from_menu(True)
        launcher.toggle_layout_mode.assert_called_with(True)
        assert "btn_modify_layout" not in launcher.__dict__

        launcher._toggle_context_help(True)
        launcher.context_help.show.assert_called_once()
        launcher._toggle_context_help(False)
        launcher.context_help.hide.assert_called_once()


def test_cleanup_processes(qapp) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        proc1 = MagicMock()
        proc1.poll.return_value = 0  # Finished
        proc2 = MagicMock()
        proc2.poll.return_value = None  # Running

        launcher.running_processes = {"p1": proc1, "p2": proc2}
        launcher._cleanup_processes()
        assert "p1" not in launcher.running_processes
        assert "p2" in launcher.running_processes

        proc2.poll.return_value = 0
        launcher._cleanup_processes()
        assert not launcher.running_processes
        assert launcher.lbl_status.text() == "Ready"


def test_on_cleanup_finished_updates_running_processes(qapp):
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        launcher.running_processes = {
            "p1": MagicMock(),
            "p2": MagicMock(),
        }

        launcher._on_cleanup_finished(["p1"])

        assert "p1" not in launcher.running_processes
        assert "p2" in launcher.running_processes


@patch("src.launchers.launcher_dialogs.ThemedModalDialog")
@patch("src.launchers.upstream_drift_launcher.kill_process_tree")
def test_close_event(mock_kill, mock_dialog_cls, qapp) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        # Disconnect cleanups to prevent PyQt crashes with mocks
        launcher.docker_checker = None
        launcher.cleanup_timer = None

        launcher._save_layout = MagicMock()

        # No running processes
        from PyQt6.QtGui import QCloseEvent

        event = QCloseEvent()
        launcher.closeEvent(event)
        launcher._save_layout.assert_called()

        # With running processes, click Yes
        proc = MagicMock()
        proc.poll.return_value = None
        launcher.running_processes = {"p1": proc}
        mock_dialog_cls.return_value.exec.return_value = QDialog.DialogCode.Accepted
        mock_kill.return_value = True

        event = QCloseEvent()
        launcher.closeEvent(event)
        mock_kill.assert_called()

        # Click No
        mock_dialog_cls.return_value.exec.return_value = QDialog.DialogCode.Rejected
        event = QCloseEvent()
        launcher.closeEvent(event)
        assert not event.isAccepted()


@patch("src.launchers.launcher_dialogs.ThemedModalDialog")
def test_close_event_excludes_background_api(mock_dialog_cls, qapp) -> None:
    with patch_launcher_ui():
        launcher = UpstreamDriftLauncher()
        launcher.docker_checker = None
        launcher.cleanup_timer = None
        launcher._save_layout = MagicMock()
        proc = MagicMock()
        proc.poll.return_value = None
        launcher.running_processes = {"background_api_server": proc}
        from PyQt6.QtGui import QCloseEvent

        event = QCloseEvent()
        launcher.closeEvent(event)
        mock_dialog_cls.assert_not_called()
        assert event.isAccepted()


@patch("src.launchers.upstream_drift_launcher_main.QApplication")
@patch("src.launchers.upstream_drift_launcher_main.AsyncStartupWorker")
@patch("src.launchers.upstream_drift_launcher_main.sys.exit")
@patch("src.launchers.upstream_drift_launcher_main._install_global_ui_zoom")
@patch("src.launchers.upstream_drift_launcher_main.SplashScreen")
@patch("src.launchers.upstream_drift_launcher.UpstreamDriftLauncher")
def test_upstream_drift_launcher_main(
    _mock_launcher,
    mock_splash_cls,
    _mock_zoom,
    mock_exit,
    mock_worker,
    mock_app,
) -> None:
    mock_app.return_value.exec.return_value = 0
    mock_splash = mock_splash_cls.return_value
    mock_worker_instance = mock_worker.return_value

    with (
        patch("src.launchers.upstream_drift_launcher_main.QIcon"),
        patch("src.launchers.upstream_drift_launcher_main.ASSETS_DIR"),
    ):
        main()
        mock_app.assert_called()
        mock_worker.assert_called()
        mock_exit.assert_called()

    progress_callback = mock_worker_instance.progress_signal.connect.call_args[0][0]
    progress_callback("Loading model registry...", 10)
    mock_splash.show_message.assert_called_once_with("Loading model registry...", 10)
