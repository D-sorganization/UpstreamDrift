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


@patch.dict("sys.modules", {"mujoco": MagicMock(), "mujoco.viewer": MagicMock()})
def test_launch_generic_mjcf(launcher) -> None:
    with patch("src.launchers.launcher_simulation.Path.exists", return_value=True):
        process = MagicMock()
        launcher.process_manager.launch_script.return_value = process
        launcher._launch_generic_mjcf(Path("test.xml"))
        launcher.show_toast.assert_called_with("Launched Passive Viewer", "success")

        launcher.process_manager.launch_script.return_value = None
        with pytest.raises(RuntimeError):
            launcher._launch_generic_mjcf(Path("test.xml"))

    with patch("src.launchers.launcher_simulation.Path.exists", return_value=False):
        launcher._launch_generic_mjcf(Path("test.xml"))
        sys.modules["mujoco"].viewer.launch.assert_called_once()

    # Exception
    with patch("src.launchers.launcher_simulation.Path.exists", return_value=False):
        sys.modules["mujoco"].viewer.launch.side_effect = RuntimeError("Crash")
        with pytest.raises(RuntimeError):
            launcher._launch_generic_mjcf(Path("test.xml"))
