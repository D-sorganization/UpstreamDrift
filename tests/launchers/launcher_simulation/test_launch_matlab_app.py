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


@patch("src.launchers.launcher_simulation.secure_popen")
def test_launch_matlab_app(mock_popen, launcher) -> None:
    model = DummyModel("m2", "M2", "matlab_app", path="test.slx")
    mock_popen.return_value = MagicMock()

    launcher._launch_matlab_app(model)
    mock_popen.assert_called_once()

    # Path missing
    model.path = None
    launcher._launch_matlab_app(model)
    launcher.show_toast.assert_called_with("Invalid MATLAB configuration.", "error")

    # .bat script
    model.path = "test.bat"
    launcher._launch_matlab_app(model)
    assert mock_popen.call_count == 2

    # .m script
    model.path = "test.m"
    launcher._launch_matlab_app(model)
    assert mock_popen.call_count == 3

    # other script
    model.path = "test.txt"
    launcher._launch_matlab_app(model)
    assert mock_popen.call_count == 4

    # Exception
    model.path = "test.slx"
    mock_popen.side_effect = PermissionError("test")
    launcher._launch_matlab_app(model)
    launcher.show_toast.assert_called_with("Launch failed: test", "error")

    # Missing mathlab not found error
    mock_popen.side_effect = FileNotFoundError("test")
    launcher._launch_matlab_app(model)
    launcher.show_toast.assert_called_with(
        "MATLAB executable not found in PATH.", "error"
    )
