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

    def _get_model(self, model_id):
        return self.models.get(model_id)


@pytest.fixture
def launcher(qapp):
    return DummyLauncher()


def test_get_subprocess_env(launcher):
    with patch("os.environ.copy", return_value={"PYTHONPATH": "old/path"}):
        env = launcher._get_subprocess_env()
        assert "PYTHONPATH" in env
        assert env["MUJOCO_PLUGIN_PATH"] == ""


@patch("src.launchers.launcher_simulation.subprocess.run")
def test_check_module_dependencies(mock_run, launcher):
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


@patch("src.launchers.launcher_simulation.QMessageBox.warning")
def test_show_dependency_error(mock_warning, launcher):
    launcher._show_dependency_error("m1", "DLL load failed")
    mock_warning.assert_called_once()

    launcher._show_dependency_error("m1", "ImportError: module not found")
    assert mock_warning.call_count == 2


def test_try_launch_special_app(launcher):
    with patch.object(launcher, "_launch_urdf_generator") as mock_urdf:
        assert launcher._try_launch_special_app("urdf_generator") is True
        mock_urdf.assert_called_once()

    with patch.object(launcher, "_launch_c3d_viewer") as mock_c3d:
        assert launcher._try_launch_special_app("c3d_viewer") is True
        mock_c3d.assert_called_once()

    with patch.object(launcher, "_launch_shot_tracer") as mock_shot:
        assert launcher._try_launch_special_app("shot_tracer") is True
        mock_shot.assert_called_once()

    assert launcher._try_launch_special_app("normal_model") is False


def test_try_launch_docker(launcher):
    launcher.chk_docker.isChecked.return_value = True
    launcher.docker_available = True

    model = DummyModel("m1", "M1", "mjcf", path="test.xml")
    with patch.object(launcher, "_launch_docker_container") as mock_launch:
        assert launcher._try_launch_docker(model) is True
        mock_launch.assert_called_once()

    # Model missing path
    model = DummyModel("m1", "M1", "mjcf")
    launcher._try_launch_docker(model)
    launcher.show_toast.assert_called_with(
        "Model path missing for Docker launch.", "error"
    )

    # Exception
    model = DummyModel("m1", "M1", "mjcf", path="test.xml")
    with patch.object(launcher, "_launch_docker_container", side_effect=OSError("err")):
        launcher._try_launch_docker(model)
        launcher.show_toast.assert_called()


@patch("src.launchers.launcher_simulation.QMessageBox.question")
def test_check_local_dependencies(mock_question, launcher):
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


def test_execute_local_launch(launcher):
    model = DummyModel("m1", "M1", "mjcf", path="test.xml")

    handler = MagicMock()
    handler.launch.return_value = True
    launcher.model_handler_registry.get_handler.return_value = handler

    launcher._execute_local_launch(model)
    launcher.show_toast.assert_called_with("M1 Launched", "success")

    handler.launch.return_value = False
    launcher._execute_local_launch(model)
    launcher.show_toast.assert_called_with("Failed to launch M1", "error")

    # Handler missing
    launcher.model_handler_registry.get_handler.return_value = None
    with patch.object(launcher, "_launch_generic_mjcf") as mock_mjcf:
        launcher._execute_local_launch(model)
        mock_mjcf.assert_called_once()

    # Unknown type
    model2 = DummyModel("m3", "M3", "unknown_type", path="test.xml")
    model2.path = "test.txt"  # Not .xml
    launcher._execute_local_launch(model2)
    launcher.show_toast.assert_called_with(
        "Unknown launch type: unknown_type", "warning"
    )

    # Missing path
    model3 = DummyModel("m4", "M4", "type")
    launcher._execute_local_launch(model3)
    launcher.show_toast.assert_called_with("Model path missing.", "error")


def test_launch_simulation(launcher):
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


@patch.dict("sys.modules", {"mujoco": MagicMock(), "mujoco.viewer": MagicMock()})
def test_launch_generic_mjcf(launcher):
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


@patch("src.launchers.launcher_process_manager.start_vcxsrv")
@patch("src.launchers.launcher_simulation.QMessageBox.warning")
@patch("src.launchers.launcher_simulation.QMessageBox.critical")
def test_launch_docker_container(mock_crit, mock_warn, mock_start, launcher):
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


def test_launch_script_process(launcher):
    # WSL mode
    launcher.chk_wsl.isChecked.return_value = True
    launcher.process_manager.launch_in_wsl.return_value = True
    launcher._launch_script_process("name", Path("script.py"), Path("cwd"))
    launcher.show_toast.assert_called_with("name Launched in WSL", "success")

    launcher.process_manager.launch_in_wsl.return_value = False
    with patch("src.launchers.launcher_simulation.QMessageBox.critical") as mock_crit:
        launcher._launch_script_process("name", Path("script.py"), Path("cwd"))
        mock_crit.assert_called_once()

    # Local mode
    launcher.chk_wsl.isChecked.return_value = False
    launcher.process_manager.launch_script.return_value = MagicMock()
    launcher._launch_script_process("name", Path("script.py"), Path("cwd"))
    launcher.show_toast.assert_called_with("name Launched", "success")

    # Local mode fail
    launcher.process_manager.launch_script.return_value = None
    with patch("src.launchers.launcher_simulation.QMessageBox.critical") as mock_crit:
        launcher._launch_script_process("name", Path("script.py"), Path("cwd"))
        mock_crit.assert_called_once()


def test_launch_module_process(launcher):
    launcher.chk_wsl.isChecked.return_value = True
    launcher.process_manager.launch_module_in_wsl.return_value = True
    launcher._launch_module_process("name", "mod", Path("cwd"))
    launcher.show_toast.assert_called_with("name Launched in WSL", "success")

    launcher.chk_wsl.isChecked.return_value = False
    launcher.process_manager.launch_module.return_value = MagicMock()
    launcher._launch_module_process("name", "mod", Path("cwd"))
    launcher.show_toast.assert_called_with("name Launched", "success")

    # Local mode fail
    launcher.process_manager.launch_module.return_value = None
    with patch("src.launchers.launcher_simulation.QMessageBox.critical") as mock_crit:
        launcher._launch_module_process("name", "mod", Path("cwd"))
        mock_crit.assert_called_once()


def test_launch_urdf_generator(launcher):
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


def test_launch_c3d_viewer(launcher):
    with patch("src.launchers.launcher_simulation.Path.exists", return_value=True):
        launcher.process_manager.launch_script.return_value = MagicMock()
        launcher._launch_c3d_viewer()
        launcher.show_toast.assert_called_with("C3D Viewer launched.", "success")

        proc = MagicMock()
        proc.poll.return_value = None
        launcher.running_processes["c3d_viewer"] = proc
        launcher._launch_c3d_viewer()
        launcher.show_toast.assert_called_with(
            "C3D Viewer is already running.", "warning"
        )

    with patch("src.launchers.launcher_simulation.Path.exists", return_value=False):
        launcher.running_processes.pop("c3d_viewer", None)
        launcher._launch_c3d_viewer()
        launcher.show_toast.assert_called_with("C3D Viewer script not found.", "error")


def test_launch_shot_tracer(launcher):
    with patch("src.launchers.launcher_simulation.Path.exists", return_value=True):
        launcher.process_manager.launch_script.return_value = MagicMock()
        launcher._launch_shot_tracer()
        launcher.show_toast.assert_called_with("Shot Tracer launched.", "success")

        proc = MagicMock()
        proc.poll.return_value = None
        launcher.running_processes["shot_tracer"] = proc
        launcher._launch_shot_tracer()
        launcher.show_toast.assert_called_with(
            "Shot Tracer is already running.", "warning"
        )

    with patch("src.launchers.launcher_simulation.Path.exists", return_value=False):
        launcher._launch_shot_tracer()
        launcher.show_toast.assert_called_with("Shot Tracer script not found.", "error")


@patch("src.launchers.launcher_simulation.secure_popen")
def test_launch_matlab_app(mock_popen, launcher):
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
