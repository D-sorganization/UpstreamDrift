"""
Unit tests for GolfLauncher GUI logic (Model selection, Launching).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestGolfLauncherLogic:
    @pytest.fixture(autouse=True)
    def mock_process_manager(self):
        with patch("src.launchers.golf_launcher.ProcessManager") as mock_pm:
            mock_pm.return_value.running_processes = {}
            yield mock_pm

    @patch("src.launchers.golf_launcher.DockerCheckThread")
    def test_initialization(self, mock_thread, qtbot):
        from src.launchers.golf_launcher import GolfLauncher

        thread_instance = mock_thread.return_value
        thread_instance.result = MagicMock()

        launcher = GolfLauncher()
        qtbot.addWidget(launcher)

        launcher.engine_manager = MagicMock()
        launcher.btn_launch.setEnabled(False)

        assert "UpstreamDrift" in launcher.windowTitle(), (
            "Assertion failed: UpstreamDrift in launcher.windowTitle()"
        )
        mock_thread.return_value.start.assert_called_once()

        assert hasattr(launcher, "grid_layout"), (
            "Assertion failed: hasattr(launcher, grid_layout)"
        )
        assert hasattr(launcher, "btn_launch"), (
            "Assertion failed: hasattr(launcher, btn_launch)"
        )

    @patch("src.launchers.golf_launcher.DockerCheckThread")
    def test_model_selection_updates_ui(self, mock_thread, qtbot):
        from src.launchers.golf_launcher import GolfLauncher

        launcher = GolfLauncher()
        qtbot.addWidget(launcher)

        mock_model = SimpleNamespace(
            name="Test Model", description="Desc", id="test_model", type="mujoco"
        )

        launcher.registry = MagicMock()
        launcher.registry.get_all_models.return_value = [mock_model]
        launcher.registry.get_model.return_value = mock_model
        launcher._build_available_models()

        launcher.engine_manager = MagicMock()
        launcher.btn_launch.setEnabled(False)

        assert launcher.btn_launch.isEnabled() is False, (
            "Assertion failed: launcher.btn_launch.isEnabled() is False"
        )

        launcher.on_docker_check_complete(True)
        assert launcher.docker_available is True, (
            "Assertion failed: launcher.docker_available is True"
        )

        launcher.selected_model = None
        launcher.btn_launch.setEnabled(False)
        launcher.btn_launch.setText("SELECT A MODEL")

        launcher.select_model("test_model")

        assert launcher.selected_model == "test_model", (
            "Assertion failed: launcher.selected_model == test_model"
        )
        assert launcher.btn_launch.isEnabled() is True, (
            "Assertion failed: launcher.btn_launch.isEnabled() is True"
        )
        assert mock_model.name.upper() in launcher.btn_launch.text().upper(), (
            "Assertion failed: mock_model.name.upper() in launcher.btn_launch.text().upper()"
        )

    @patch("src.launchers.golf_launcher.DockerCheckThread")
    def test_launch_simulation_constructs_command(self, mock_thread, qtbot):
        from src.launchers.golf_launcher import GolfLauncher

        launcher = GolfLauncher()
        qtbot.addWidget(launcher)

        mock_model = SimpleNamespace(
            name="Test Model", path="engines/test", id="test_model", type="docker"
        )
        launcher.registry = MagicMock()
        launcher.registry.get_all_models.return_value = [mock_model]
        launcher.registry.get_model.return_value = mock_model
        launcher._build_available_models()

        launcher.engine_manager = MagicMock()
        launcher.btn_launch.setEnabled(False)
        launcher.docker_available = True

        # Patch docker_launcher
        launcher.docker_launcher = MagicMock()
        launcher.docker_launcher.check_image_exists.return_value = True
        launcher.docker_launcher.launch_container.return_value = MagicMock()

        # Check docker requires setting the actual checkbox
        launcher.chk_docker.setChecked(True)

        launcher.select_model("test_model")

        with (
            patch.object(Path, "exists", return_value=True),
            patch(
                "src.launchers.launcher_simulation.resolve_model_artifact_path",
                return_value=Path("engines/test"),
            ),
        ):
            launcher.launch_simulation()

        launcher.docker_launcher.launch_container.assert_called_once()
        args, kwargs = launcher.docker_launcher.launch_container.call_args
        assert kwargs["model_type"] == "docker", (
            "Assertion failed: kwargs[model_type] == docker"
        )
        assert kwargs["model_name"] == "Test Model", (
            "Assertion failed: kwargs[model_name] == Test Model"
        )

    @patch("src.launchers.golf_launcher.DockerCheckThread")
    def test_launch_generic_mjcf(self, mock_thread, qtbot):
        from src.launchers.golf_launcher import GolfLauncher

        launcher = GolfLauncher()
        qtbot.addWidget(launcher)

        mock_model = SimpleNamespace(
            name="Generic MJCF",
            path="engines/test/model.xml",
            id="generic_mjcf",
            type="mjcf",
        )
        launcher.registry = MagicMock()
        launcher.registry.get_all_models.return_value = [mock_model]
        launcher.registry.get_model.return_value = mock_model
        launcher._build_available_models()

        launcher.engine_manager = MagicMock()
        launcher.btn_launch.setEnabled(False)
        launcher.docker_available = True
        launcher.select_model("generic_mjcf")

        # Fake check local dependencies
        launcher._check_local_dependencies = MagicMock(return_value=True)

        mock_mujoco = MagicMock()
        mock_viewer = MagicMock()

        with (
            patch.dict(
                "sys.modules", {"mujoco": mock_mujoco, "mujoco.viewer": mock_viewer}
            ),
            patch("src.launchers.launcher_simulation.Path.exists", return_value=False),
            patch(
                "src.launchers.launcher_simulation.resolve_model_artifact_path",
                return_value=Path("engines/test/model.xml"),
            ),
            patch(
                "src.launchers.launcher_model_handlers.ModelHandlerRegistry.get_handler",
                return_value=None,
            ),
        ):
            launcher.launch_simulation()

            mock_mujoco.MjModel.from_xml_path.assert_called_once()
            mock_mujoco.viewer.launch.assert_called_once()

    @patch("src.launchers.golf_launcher.DockerCheckThread")
    def test_launch_matlab_suite(self, mock_thread, qtbot):
        from src.launchers.golf_launcher import GolfLauncher

        launcher = GolfLauncher()
        qtbot.addWidget(launcher)

        mock_model = SimpleNamespace(
            name="Matlab Suite",
            path="virtual/matlab_suite",
            id="matlab_suite",
            type="matlab_suite",
        )
        launcher.registry = MagicMock()
        launcher.registry.get_all_models.return_value = [mock_model]
        launcher.registry.get_model.return_value = mock_model
        launcher._build_available_models()
        launcher.select_model("matlab_suite")

        with patch(
            "src.launchers.matlab_suite_dialog.MatlabSuiteDialog"
        ) as mock_dialog:
            launcher.launch_simulation()
            mock_dialog.assert_called_once_with(launcher)
            mock_dialog.return_value.exec.assert_called_once()
