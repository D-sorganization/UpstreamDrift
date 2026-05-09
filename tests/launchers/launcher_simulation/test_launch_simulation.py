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


def test_launch_simulation(launcher) -> None:
    # No selected model
    launcher.launch_simulation()

    # Special app
    launcher.selected_model = "urdf_generator"
    with patch.object(launcher, "_try_launch_special_app", return_value=True):
        launcher.launch_simulation()

    # Missing model configuration
    launcher.selected_model = "m99"
    launcher.launch_simulation()
    launcher.show_toast.assert_called_with("Model configuration not found.", "error")

    # Matlab app
    launcher.selected_model = "m2"
    with patch.object(launcher, "_launch_matlab_app") as mock_matlab:
        launcher.launch_simulation()
        mock_matlab.assert_called_once()

    # Docker launch
    launcher.selected_model = "m1"
    with patch.object(launcher, "_try_launch_docker", return_value=True):
        launcher.launch_simulation()

    # Execute local
    with (
        patch.object(launcher, "_try_launch_docker", return_value=False),
        patch.object(launcher, "_check_local_dependencies", return_value=True),
        patch.object(launcher, "_execute_local_launch") as mock_exec,
    ):
        launcher.launch_simulation()
        mock_exec.assert_called_once()

    # Execute local error
    with (
        patch.object(launcher, "_try_launch_docker", return_value=False),
        patch.object(launcher, "_check_local_dependencies", return_value=True),
        patch.object(launcher, "_execute_local_launch", side_effect=ValueError("Test")),
    ):
        launcher.launch_simulation()
        launcher.show_toast.assert_called_with("Launch Failed: Test", "error")
