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


@patch("src.launchers.launcher_simulation.subprocess.run")
def test_check_module_dependencies(mock_run, launcher) -> None:
    mock_run.return_value.stdout = "OK"
    success, err = launcher._check_module_dependencies("mjcf")
    assert success is True

    mock_run.return_value.stdout = "ImportError: no module"
    success, err = launcher._check_module_dependencies("mjcf")
    assert success is False
    assert "dependency check failed" in err

    # Timeout
    import subprocess

    mock_run.side_effect = subprocess.TimeoutExpired("cmd", 10)
    success, err = launcher._check_module_dependencies("drake")
    assert success is False

    # OS Error
    mock_run.side_effect = OSError("failed")
    success, err = launcher._check_module_dependencies("pinocchio")
    assert success is False

    # Unknown type
    success, err = launcher._check_module_dependencies("not_a_real_type")
    assert success is True
