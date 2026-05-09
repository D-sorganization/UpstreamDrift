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


def test_launch_urdf_generator(launcher) -> None:
    with patch("src.launchers.launcher_simulation.Path.exists", return_value=True):
        launcher.process_manager.launch_script.return_value = MagicMock()
        launcher._launch_urdf_generator()
        launcher.show_toast.assert_called_with("URDF Generator launched.", "success")

        # Already running
        proc = MagicMock()
        proc.poll.return_value = None
        launcher.running_processes["urdf_generator"] = proc
        launcher._launch_urdf_generator()
        launcher.show_toast.assert_called_with(
            "URDF Generator is already running.", "warning"
        )

        launcher.running_processes.pop("urdf_generator", None)
        launcher.process_manager.launch_script.return_value = None
        launcher._launch_urdf_generator()
        launcher.show_toast.assert_called_with(
            "Launch failed: ProcessManager returned None", "error"
        )
