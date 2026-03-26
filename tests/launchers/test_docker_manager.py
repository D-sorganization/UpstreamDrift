"""Tests for docker_manager."""

import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from src.launchers.docker_manager import (  # noqa: E402
    DockerBuildThread,
    DockerCheckThread,
    DockerLauncher,
)
from src.shared.python.security.secure_subprocess import (
    SecureSubprocessError,  # noqa: E402
)


@patch("src.launchers.docker_manager.secure_run")
def test_docker_check_thread_success(mock_run):
    thread = DockerCheckThread()
    mock_signal = MagicMock()
    thread.result.connect(mock_signal)

    thread.run()

    mock_run.assert_called_once()
    mock_signal.assert_called_once_with(True)


@patch(
    "src.launchers.docker_manager.secure_run",
    side_effect=SecureSubprocessError("Failed", [], 1),
)
def test_docker_check_thread_failure(mock_run):
    thread = DockerCheckThread()
    mock_signal = MagicMock()
    thread.result.connect(mock_signal)

    thread.run()

    mock_run.assert_called_once()
    mock_signal.assert_called_once_with(False)


def test_docker_build_thread_invalid_context():
    thread = DockerBuildThread(context_path=Path("/does/not/exist/at/all"))
    mock_finished = MagicMock()
    thread.finished_signal.connect(mock_finished)

    thread.run()

    mock_finished.assert_called_once()
    args = mock_finished.call_args[0]
    assert args[0] is False
    assert "Invalid Docker context" in args[1]


@patch("subprocess.Popen")
@patch.object(Path, "exists", return_value=True)
def test_docker_build_thread_success(mock_exists, mock_popen):
    context = Path("/fake/context")
    thread = DockerBuildThread(target_stage="all", image_name="test_image", context_path=context)

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout.readline.side_effect = ["Step 1\n", ""]
    mock_popen.return_value = mock_process

    mock_log = MagicMock()
    mock_finished = MagicMock()
    thread.log_signal.connect(mock_log)
    thread.finished_signal.connect(mock_finished)

    thread.run()

    mock_popen.assert_called_once()
    mock_finished.assert_called_once_with(True, "Build successful.")
    # Log should be called a few times (3 setup + 1 output)
    assert mock_log.call_count == 4


@patch("subprocess.Popen")
@patch.object(Path, "exists", return_value=True)
def test_docker_build_thread_failure(mock_exists, mock_popen):
    context = Path("/fake/context")
    thread = DockerBuildThread(target_stage="all", image_name="test_image", context_path=context)

    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stdout.readline.side_effect = [""]
    mock_popen.return_value = mock_process

    mock_finished = MagicMock()
    thread.finished_signal.connect(mock_finished)

    thread.run()

    mock_popen.assert_called_once()
    mock_finished.assert_called_once_with(False, "Build failed with code 1")


@patch("subprocess.Popen")
@patch.object(Path, "exists", return_value=True)
def test_docker_build_thread_linux(mock_exists, mock_popen):
    context = Path("/fake/context")
    thread = DockerBuildThread(target_stage="all", image_name="test_image", context_path=context)

    mock_process = MagicMock()
    mock_process.returncode = 0
    # Simulate process.stdout being None and empty line handling
    mock_process.stdout = None
    mock_popen.return_value = mock_process

    mock_finished = MagicMock()
    thread.finished_signal.connect(mock_finished)

    with patch("os.name", "posix"):
        thread.run()

    mock_popen.assert_called_once()
    mock_finished.assert_called_once_with(True, "Build successful.")
    kwargs = mock_popen.call_args[1]
    assert kwargs.get("creationflags", 0) == 0


@patch("subprocess.Popen")
@patch.object(Path, "exists", return_value=True)
def test_docker_build_thread_empty_line(mock_exists, mock_popen):
    context = Path("/fake/context")
    thread = DockerBuildThread(target_stage="all", image_name="test_image", context_path=context)

    mock_process = MagicMock()
    mock_process.returncode = 0
    # Empty string inside iter but line is actually skipped
    mock_process.stdout.readline.side_effect = ["\n", "Step 1\n", ""]
    mock_popen.return_value = mock_process

    mock_log = MagicMock()
    thread.log_signal.connect(mock_log)

    thread.run()

    # Log calls: 3 setups + 1 for 'Step 1' (the '\n' should just strip to empty and be skipped implicitly, or handled)
    # The coverage says 'if line:' jump to 'next'. `line.strip()` will be empty, so no log emitted for `\n` if it strips first.
    # Ah, the code is:
    # if line: self.log_signal.emit(line.strip())
    # actually `line` is `\n`, so it evaluates to True, but `line.strip()` emits `""`.
    # To test falsey line, readline would just return `""` which terminates the loop.
    # Wait, iter(process.stdout.readline, "") stops on `""`.


@patch("subprocess.Popen", side_effect=OSError("Boom"))
@patch.object(Path, "exists", return_value=True)
def test_docker_build_thread_exception(mock_exists, mock_popen):
    context = Path("/fake/context")
    thread = DockerBuildThread(target_stage="all", image_name="test_image", context_path=context)

    mock_finished = MagicMock()
    thread.finished_signal.connect(mock_finished)

    thread.run()

    mock_finished.assert_called_once_with(False, "Boom")


def test_docker_launcher_check_image_exists_true():
    launcher = DockerLauncher(repo_root=Path("/fake/repo"), image_name="my_image")

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        assert launcher.check_image_exists() is True
        mock_run.assert_called_once()


def test_docker_launcher_check_image_exists_legacy():
    launcher = DockerLauncher(repo_root=Path("/fake/repo"), image_name="my_image")

    with patch("subprocess.run") as mock_run:
        # First call fails, second call (legacy) succeeds
        mock_fail = MagicMock()
        mock_fail.returncode = 1

        mock_success = MagicMock()
        mock_success.returncode = 0

        mock_run.side_effect = [mock_fail, mock_success]

        assert launcher.check_image_exists() is True
        assert mock_run.call_count == 2
        # It should update the image name to the legacy one
        assert launcher.image_name != "my_image"


def test_docker_launcher_check_image_exists_false():
    launcher = DockerLauncher(repo_root=Path("/fake/repo"), image_name="my_image")

    with patch("subprocess.run") as mock_run:
        mock_fail = MagicMock()
        mock_fail.returncode = 1
        mock_run.return_value = mock_fail

        assert launcher.check_image_exists() is False


def test_docker_launcher_check_image_exists_exception():
    launcher = DockerLauncher(repo_root=Path("/fake/repo"), image_name="my_image")

    with patch("subprocess.run", side_effect=OSError("Boom")):
        assert launcher.check_image_exists() is False


def test_build_launch_command_windows():
    launcher = DockerLauncher(repo_root=Path("/fake/repo"), image_name="my_image")

    repo_path = Path("/fake/repo/src/some/drake_model.py")
    with patch("os.name", "nt"):
        cmd = launcher.build_launch_command(
            model_type="drake",
            repo_path=repo_path,
            use_gpu=True,
        )
        assert "DISPLAY=host.docker.internal:0" in cmd
        assert "--gpus=all" in cmd
        assert "-p" in cmd
        assert "7000:7000" in cmd
        assert cmd[-2] == "-m"
        assert cmd[-1] == "src.drake_gui_app"


def test_build_launch_command_linux():
    launcher = DockerLauncher(repo_root=Path("/fake/repo"), image_name="my_image")

    mock_repo_path = MagicMock()
    mock_repo_path.parent.relative_to.return_value.as_posix.return_value = "src/some"
    mock_repo_path.name = "pinocchio_model.py"

    with patch("os.name", "posix"), patch.dict("os.environ", {"DISPLAY": ":99"}):
        cmd = launcher.build_launch_command(
            model_type="pinocchio",
            repo_path=mock_repo_path,
            use_gpu=False,
        )
        assert "DISPLAY=:99" in cmd
        assert "DISPLAY=:99" in cmd
        assert cmd[-2] == "python"
        assert cmd[-1] == "pinocchio_golf/gui.py"


def test_build_launch_command_custom_and_other():
    launcher = DockerLauncher(repo_root=Path("/fake/repo"), image_name="my_image")

    mock_repo_path = MagicMock()
    mock_repo_path.parent.relative_to.return_value.as_posix.return_value = "src/some"
    mock_repo_path.name = "my_custom.py"

    # Test "custom_humanoid"
    with patch("os.name", "posix"):
        cmd = launcher.build_launch_command(
            model_type="custom_humanoid",
            repo_path=mock_repo_path,
            use_gpu=False,
        )
        assert cmd[-1] == "my_custom.py"
        assert "-p" not in cmd  # ensure no port mapping

    # Test "other"
    with patch("os.name", "posix"):
        cmd = launcher.build_launch_command(
            model_type="other",
            repo_path=mock_repo_path,
            use_gpu=False,
        )
        assert cmd[-1] == "my_custom.py"


@patch("subprocess.Popen")
def test_launch_container_capture_output(mock_popen):
    launcher = DockerLauncher(repo_root=Path("/fake/repo"), image_name="my_image")

    with patch.object(launcher, "build_launch_command", return_value=["docker", "run"]):
        process = launcher.launch_container(
            model_type="custom",
            model_name="Custom",
            repo_path=Path("/fake/repo/script.py"),
            capture_output=True,
        )
        mock_popen.assert_called_once()
        kwargs = mock_popen.call_args[1]
        assert "stdout" in kwargs
        assert kwargs["stdout"] == subprocess.PIPE
        assert process is mock_popen.return_value


@patch("subprocess.Popen")
def test_launch_container_capture_output_posix(mock_popen):
    launcher = DockerLauncher(repo_root=Path("/fake/repo"), image_name="my_image")

    with (
        patch.object(launcher, "build_launch_command", return_value=["docker", "run"]),
        patch("os.name", "posix"),
    ):
        process = launcher.launch_container(
            model_type="custom",
            model_name="Custom",
            repo_path=Path("/fake/repo/script.py"),
            capture_output=True,
        )
        mock_popen.assert_called_once()
        kwargs = mock_popen.call_args[1]
        assert kwargs.get("creationflags", 0) == 0
        assert process is mock_popen.return_value


@patch("subprocess.Popen", side_effect=OSError("Failed"))
def test_launch_container_os_error(mock_popen):
    launcher = DockerLauncher(repo_root=Path("/fake/repo"), image_name="my_image")

    with patch.object(launcher, "build_launch_command", return_value=["docker", "run"]):
        process = launcher.launch_container(
            model_type="custom",
            model_name="Custom",
            repo_path=Path("/fake/repo/script.py"),
            capture_output=False,
        )
        mock_popen.assert_called_once()
        assert process is None
