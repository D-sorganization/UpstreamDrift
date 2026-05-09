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


def test_launch_c3d_viewer_all_candidates_missing_logs_full_search_list(
    launcher, caplog
) -> None:
    """Pins #4595: when no candidate exists, log the full search list and toast.

    The function must report ``C3D Viewer script not found`` AND log every
    path it searched, not just the first miss.
    """
    import logging as _logging

    with patch("src.launchers.launcher_simulation.Path.exists", return_value=False):
        launcher.running_processes.pop("c3d_viewer", None)
        with caplog.at_level(
            _logging.ERROR, logger="src.launchers.launcher_simulation"
        ):
            launcher._launch_c3d_viewer()
    launcher.show_toast.assert_called_with("C3D Viewer script not found.", "error")
    not_found = [
        r for r in caplog.records if "C3D Viewer script not found" in r.getMessage()
    ]
    assert not_found, "expected an ERROR-level log entry"
    msg = not_found[0].getMessage()
    assert "run_c3d_viewer.py" in msg
    assert "launch_pyqt6.py" in msg
    assert "c3d_viewer.py" in msg
