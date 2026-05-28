"""
Unit tests for UpstreamDriftLauncher GUI logic (Model selection, Launching).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestUpstreamDriftLauncherLogic:
    @pytest.fixture(autouse=True)
    def mock_process_manager(self):
        with patch("src.launchers.upstream_drift_launcher.ProcessManager") as mock_pm:
            mock_pm.return_value.running_processes = {}
            yield mock_pm

    @patch("src.launchers.upstream_drift_launcher.DockerCheckThread")
    def test_upstream_drift_launcher_logic_initialization(self, mock_thread, qtbot):
        from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

        thread_instance = mock_thread.return_value
        thread_instance.result = MagicMock()

        launcher = UpstreamDriftLauncher()
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

    @patch("src.launchers.upstream_drift_launcher.DockerCheckThread")
    def test_model_selection_updates_ui(self, mock_thread, qtbot):
        from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

        launcher = UpstreamDriftLauncher()
        qtbot.addWidget(launcher)

        mock_model = SimpleNamespace(
            name="Test Model", description="Desc", id="test_model", type="mujoco"
        )

        launcher.registry = MagicMock()
        launcher.registry.get_all_models.return_value = [mock_model]
        launcher.registry.get_model.return_value = mock_model
        launcher.orchestrator.build_available_models()

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

    @patch("src.launchers.upstream_drift_launcher.DockerCheckThread")
    def test_launch_simulation_constructs_command(self, mock_thread, qtbot):
        from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

        launcher = UpstreamDriftLauncher()
        qtbot.addWidget(launcher)

        mock_model = SimpleNamespace(
            name="Test Model", path="engines/test", id="test_model", type="docker"
        )
        launcher.registry = MagicMock()
        launcher.registry.get_all_models.return_value = [mock_model]
        launcher.registry.get_model.return_value = mock_model
        launcher.orchestrator.build_available_models()

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

    @patch("src.launchers.upstream_drift_launcher.DockerCheckThread")
    def test_launch_generic_mjcf(self, mock_thread, qtbot):
        from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

        launcher = UpstreamDriftLauncher()
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
        launcher.orchestrator.build_available_models()

        launcher.engine_manager = MagicMock()
        launcher.btn_launch.setEnabled(False)
        launcher.docker_available = False
        launcher.chk_docker.setChecked(False)
        launcher.select_model("generic_mjcf")

        # Fake check local dependencies
        launcher._check_local_dependencies = MagicMock(return_value=True)  # type: ignore[attr-defined]

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

    @patch("src.launchers.upstream_drift_launcher.DockerCheckThread")
    def test_launch_matlab_suite(self, mock_thread, qtbot):
        from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

        launcher = UpstreamDriftLauncher()
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
        launcher.orchestrator.build_available_models()
        launcher.select_model("matlab_suite")

        with (
            patch.object(launcher, "dock_widget_as_tab") as mock_dock,
            patch("src.launchers.matlab_suite_dialog.MatlabSuiteWidget") as mock_widget,
        ):
            launcher.launch_simulation()
            mock_widget.assert_called_once_with(launcher)
            mock_dock.assert_called_once_with(mock_widget.return_value, "Matlab Suite")
