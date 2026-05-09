"""Tests for launcher_simulation.py."""

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtWidgets import QMainWindow, QMessageBox  # noqa: E402
from src.launchers.launcher_simulation import LauncherSimulationMixin  # noqa: E402


class DummyModel:
    def __init__(self, id, name, type, path=None):
        self.id = id
        self.name = name
        self.type = type
        self.path = path


class DummyLauncher(QMainWindow, LauncherSimulationMixin):
    def __init__(self):
        super().__init__()
        self.selected_model = None
        self.show_toast = MagicMock()
        self.lbl_status = MagicMock()
        self.chk_docker = MagicMock()
        self.chk_docker.isChecked.return_value = False
        self.chk_wsl = MagicMock()
        self.chk_wsl.isChecked.return_value = False
        self.chk_gpu = MagicMock()
        self.chk_gpu.isChecked.return_value = False
        self.docker_available = False
        self.process_manager = MagicMock()
        self.model_handler_registry = MagicMock()
        self.docker_launcher = MagicMock()
        self.running_processes = {}
        self.models = {
            "m1": DummyModel("m1", "M1", "mjcf", path="test.xml"),
            "m2": DummyModel("m2", "M2", "matlab_app", path="test.slx"),
        }

    def _get_model(self, model_id: str) -> DummyModel | None:
        return self.models.get(model_id)


@pytest.fixture
def launcher(qapp) -> DummyLauncher:
    return DummyLauncher()


@patch("src.launchers.launcher_process_manager.start_vcxsrv")
@patch("src.launchers.launcher_simulation.QMessageBox.warning")
@patch("src.launchers.launcher_simulation.QMessageBox.critical")
def test_launch_docker_container(mock_crit, mock_warn, mock_start, launcher) -> None:
    mock_start.return_value = True
    model = DummyModel("m1", "M1", "mjcf", path="test.xml")
    repo_path = Path("test.xml")

    # Image missing
    launcher.docker_launcher.check_image_exists.return_value = False
    launcher._launch_docker_container(model, repo_path)
    mock_warn.assert_called_once()

    # Launch success
    launcher.docker_launcher.check_image_exists.return_value = True
    process = MagicMock()
    launcher.docker_launcher.launch_container.return_value = process
    launcher._launch_docker_container(model, repo_path)
    launcher.process_manager.attach_process.assert_called_once()

    # Launch fail
    launcher.docker_launcher.launch_container.return_value = None
    launcher._launch_docker_container(model, repo_path)
    mock_crit.assert_called_once()

    # Exception
    launcher.docker_launcher.launch_container.side_effect = ValueError("test")
    launcher._launch_docker_container(model, repo_path)
    assert mock_crit.call_count == 2

    # Windows vcxsrv unavailable
    with patch("src.launchers.launcher_simulation.os.name", "nt"):
        mock_start.return_value = False
        with patch(
            "src.launchers.launcher_simulation.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ):
            launcher._launch_docker_container(model, repo_path)
