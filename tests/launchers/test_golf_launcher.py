"""Tests for golf_launcher.py."""

import contextlib  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtWidgets import QMessageBox  # noqa: E402

from src.launchers.golf_launcher import GolfLauncher, main  # noqa: E402
from src.launchers.ui_components import StartupResults  # noqa: E402


@pytest.fixture
def startup_results():
    results = StartupResults()
    results.startup_time_ms = 100
    results.docker_available = True
    results.registry = MagicMock()
    results.engine_manager = MagicMock()
    return results


@contextlib.contextmanager
def patch_launcher_ui():
    with (
        patch("src.launchers.golf_launcher.DockerCheckThread"),
        patch("src.launchers.golf_launcher.QTimer"),
    ):
        yield


def test_init_without_results(qapp):
    with (
        patch_launcher_ui(),
        patch("src.launchers.golf_launcher._lazy_load_model_registry") as mock_reg,
        patch("src.launchers.golf_launcher._lazy_load_engine_manager") as mock_eng,
    ):
        mock_reg.return_value = MagicMock()
        mock_eng.return_value = (MagicMock(), MagicMock())
        launcher = GolfLauncher()
        assert launcher._startup_time_ms == 0
        assert launcher.docker_available is False
        mock_reg.assert_called_once()
        mock_eng.assert_called_once()


def test_init_with_results(qapp, startup_results):
    with patch_launcher_ui():
        launcher = GolfLauncher(startup_results)
        assert launcher._startup_time_ms == 100
        assert launcher.docker_available is True
        assert launcher.registry == startup_results.registry
        assert launcher.engine_manager == startup_results.engine_manager


def test_load_window_icon(qapp):
    with patch_launcher_ui():
        # Do not mock QIcon. Just instantiate and test it doesn't crash.
        launcher = GolfLauncher()
        assert launcher.windowIcon() is not None


def test_init_registry_exception(qapp):
    with patch(
        "src.launchers.golf_launcher._lazy_load_model_registry",
        side_effect=ImportError("test"),
    ):
        launcher = GolfLauncher()
        assert launcher.registry is None


def test_init_engine_manager_exception(qapp):
    with patch(
        "src.launchers.golf_launcher._lazy_load_engine_manager",
        side_effect=RuntimeError("test"),
    ):
        launcher = GolfLauncher()
        assert launcher.engine_manager is None


def test_build_available_models(qapp, startup_results):
    model1 = MagicMock(id="m1", type="sim")
    model2 = MagicMock(id="m2", type="utility")
    startup_results.registry.get_all_models.return_value = [model1, model2]

    launcher = GolfLauncher(startup_results)
    assert "m1" in launcher.available_models
    assert "m2" in launcher.special_app_lookup


def test_get_model(qapp):
    with patch_launcher_ui():
        launcher = GolfLauncher()
        launcher.available_models["m1"] = "Model1"
        assert launcher._get_model("m1") == "Model1"

        launcher.registry = MagicMock()
        launcher.registry.get_model.return_value = "ModelX"
        assert launcher._get_model("mx") == "ModelX"

        launcher.registry = None
        assert launcher._get_model("mx") is None


def test_layout_management(qapp):
    with patch_launcher_ui():
        launcher = GolfLauncher()
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


def test_launch_model_direct(qapp):
    with patch_launcher_ui():
        launcher = GolfLauncher()
        launcher.select_model = MagicMock()
        launcher.launch_simulation = MagicMock()
        launcher.launch_model_direct("m1")
        launcher.select_model.assert_called_with("m1")
        launcher.launch_simulation.assert_called_once()


def test_center_window(qapp):
    from PyQt6.QtCore import QPoint

    with patch_launcher_ui():
        launcher = GolfLauncher()
        mock_screen = MagicMock()
        mock_geom = MagicMock()
        mock_geom.center.return_value = QPoint(0, 0)
        mock_screen.availableGeometry.return_value = mock_geom
        # To avoid type error when move is called with MagicMock
        launcher.move = MagicMock()
        launcher.screen = MagicMock(return_value=mock_screen)
        launcher.center_window()
        launcher.move.assert_called_once()


def test_load_layout_empty(qapp):
    with patch_launcher_ui():
        launcher = GolfLauncher()
        launcher.layout_manager = MagicMock()
        launcher.layout_manager.load_layout.return_value = None
        launcher._rebuild_grid = MagicMock()
        launcher._load_layout()
        launcher._rebuild_grid.assert_called_once()


def test_load_layout_with_data(qapp):
    with patch_launcher_ui():
        launcher = GolfLauncher()
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


def test_select_model(qapp):
    with patch_launcher_ui():
        launcher = GolfLauncher()
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
        card_m1.setStyleSheet.assert_called()
        card_m2.setStyleSheet.assert_called()
        launcher.update_launch_button.assert_called_once_with("Test Model")
        launcher.context_help.update_context.assert_called_with("m1")


def test_update_launch_button(qapp):
    with patch_launcher_ui():
        launcher = GolfLauncher()

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


@patch("src.launchers.golf_launcher._lazy_load_engine_manager")
def test_get_engine_type(mock_lazy_em, qapp):
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

        launcher = GolfLauncher()
        assert launcher._get_engine_type("mujoco") == "mujoco_enum"
        assert launcher._get_engine_type("drake") == "drake_enum"
        assert launcher._get_engine_type("pinocchio") == "pinocchio_enum"
        assert launcher._get_engine_type("opensim") == "opensim_enum"
        assert launcher._get_engine_type("myosim") == "myosim_enum"
        assert launcher._get_engine_type("unknown") == "mujoco_enum"


def test_docker_status(qapp):
    with patch_launcher_ui():
        launcher = GolfLauncher()
        launcher.update_launch_button = MagicMock()

        launcher._apply_docker_status(True)
        assert launcher.lbl_status.text() == "System Ready"

        launcher._apply_docker_status(False)
        assert launcher.lbl_status.text() == "Docker Not Found"


def test_check_docker(qapp):
    # patch_launcher_ui already mocks DockerCheckThread, so we don't need a separate patch block.
    # However we can just grab it off the launcher to assert on it.
    with patch_launcher_ui():
        launcher = GolfLauncher()
        # _init_ automatically calls check_docker and starts it.
        # It was mocked globally as DockerCheckThread, let's reset it and test check_docker manually.
        launcher.docker_checker.reset_mock()

        # Second time, already running
        launcher.docker_checker.isRunning.return_value = True
        launcher.check_docker()
        launcher.docker_checker.wait.assert_called()


def test_menu_toggles(qapp):
    with patch_launcher_ui():
        launcher = GolfLauncher()
        launcher.toggle_layout_mode = MagicMock()
        launcher.context_help = MagicMock()

        launcher._toggle_layout_mode_from_menu(True)
        assert launcher.btn_modify_layout.isChecked()
        launcher.toggle_layout_mode.assert_called_with(True)

        launcher._toggle_context_help(True)
        launcher.context_help.show.assert_called_once()
        launcher._toggle_context_help(False)
        launcher.context_help.hide.assert_called_once()


def test_cleanup_processes(qapp):
    with patch_launcher_ui():
        launcher = GolfLauncher()

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


@patch("src.launchers.golf_launcher.QMessageBox.question")
@patch("src.launchers.golf_launcher.kill_process_tree")
def test_close_event(mock_kill, mock_question, qapp):
    with patch_launcher_ui():
        launcher = GolfLauncher()
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
        mock_question.return_value = QMessageBox.StandardButton.Yes
        mock_kill.return_value = True

        event = QCloseEvent()
        launcher.closeEvent(event)
        mock_kill.assert_called()

        # Click No
        mock_question.return_value = QMessageBox.StandardButton.No
        event = QCloseEvent()
        launcher.closeEvent(event)
        assert not event.isAccepted()


@patch("src.launchers.golf_launcher.QApplication")
@patch("src.launchers.golf_launcher.AsyncStartupWorker")
@patch("src.launchers.golf_launcher.sys.exit")
def test_main(mock_exit, mock_worker, mock_app):
    with patch("src.launchers.golf_launcher.GolfSplashScreen"):
        main()
        mock_app.assert_called()
        mock_worker.assert_called()
        mock_exit.assert_called()
