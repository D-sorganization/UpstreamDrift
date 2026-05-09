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


@patch("src.launchers.launcher_simulation.QMessageBox.question")
def test_check_local_dependencies(mock_question, launcher) -> None:
    model = DummyModel("m1", "M1", "mujoco", path="test.xml")

    # WSL enabled
    launcher.chk_wsl.isChecked.return_value = True
    assert launcher._check_local_dependencies(model) is True

    # WSL disabled, deps OK
    launcher.chk_wsl.isChecked.return_value = False
    with patch.object(launcher, "_check_module_dependencies", return_value=(True, "")):
        assert launcher._check_local_dependencies(model) is True

    # Deps fail, docker available
    launcher.docker_available = True
    with patch.object(
        launcher, "_check_module_dependencies", return_value=(False, "error")
    ):
        mock_question.return_value = QMessageBox.StandardButton.Yes
        with patch.object(launcher, "launch_simulation") as mock_launch:
            assert launcher._check_local_dependencies(model) is False
            launcher.chk_docker.setChecked.assert_called_with(True)
            mock_launch.assert_called_once()

        mock_question.return_value = QMessageBox.StandardButton.No
        with patch.object(launcher, "_show_dependency_error"):
            assert launcher._check_local_dependencies(model) is False
