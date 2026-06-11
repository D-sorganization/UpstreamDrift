"""Regression coverage for Docker build timeout handling."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.launchers import docker_manager
from src.launchers.docker_manager import (
    DockerBuildThread,
    DockerCheckThread,
    DockerLauncher,
)

pytestmark = pytest.mark.unit


class BlockingStdout:
    """Stdout fake that remains open until closed by process cleanup."""

    def __init__(self) -> None:
        self._closed = threading.Event()

    def readline(self) -> str:
        self._closed.wait()
        return ""

    def close(self) -> None:
        self._closed.set()


class TimeoutProcess:
    """Popen fake whose stdout never reaches EOF before the process timeout."""

    pid = 4242
    returncode = None

    def __init__(self) -> None:
        self.stdout = BlockingStdout()
        self.args = ["docker", "build"]
        self.wait_timeouts: list[float | None] = []
        self.killed = False
        self.terminated = False

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        raise subprocess.TimeoutExpired(cmd=["docker", "build"], timeout=timeout)

    def terminate(self) -> None:
        self.terminated = True
        self.stdout.close()

    def kill(self) -> None:
        self.killed = True
        self.stdout.close()


class LineStdout:
    """Stdout fake that yields a finite sequence of Docker output lines."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)
        self.closed = False

    def readline(self) -> str:
        return next(self._lines, "")

    def close(self) -> None:
        self.closed = True


class CompletedProcess:
    """Popen fake for successful or failed Docker build completion."""

    pid = 5252
    args = ["docker", "build"]

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.stdout = LineStdout(["step 1\n", "step 2\n"])
        self.wait_timeouts: list[float | None] = []
        self.killed = False
        self.terminated = False

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class CancellableProcess:
    """Popen fake that exits only after the build thread cancels it."""

    pid = 6262
    args = ["docker", "build"]

    def __init__(self) -> None:
        self.returncode = None
        self.stdout = BlockingStdout()
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float | None] = []

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd=self.args, timeout=timeout)
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self.stdout.close()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.stdout.close()


@patch("subprocess.Popen")
@patch.object(Path, "exists", return_value=True)
def test_docker_build_timeout_is_enforced_while_stdout_is_open(
    _mock_exists: MagicMock,
    mock_popen: MagicMock,
) -> None:
    process = TimeoutProcess()
    mock_popen.return_value = process

    thread = DockerBuildThread(
        target_stage="all",
        image_name="test_image",
        context_path=Path("/fake/context"),
        build_timeout_seconds=0.01,
    )
    finished = MagicMock()
    thread.finished_signal.connect(finished)

    with patch("src.launchers.docker_manager.kill_process_tree") as kill_tree:

        def reap_process(_pid: int) -> bool:
            process.returncode = -9
            process.stdout.close()
            return True

        kill_tree.side_effect = reap_process

        thread.run()

    assert 0 < process.wait_timeouts[0] <= 0.02
    kill_tree.assert_called_once_with(process.pid)
    assert process.killed is False
    finished.assert_called_once_with(False, "Build timed out (exceeded 1 hour limit)")


@patch("subprocess.Popen")
@patch.object(Path, "exists", return_value=True)
def test_docker_build_cancel_terminates_child_and_emits_cancelled(
    _mock_exists: MagicMock,
    mock_popen: MagicMock,
    qapp,
) -> None:
    process = CancellableProcess()
    mock_popen.return_value = process
    thread = DockerBuildThread(
        target_stage="all",
        image_name="test_image",
        context_path=Path("/fake/context"),
    )
    finished = MagicMock()
    thread.finished_signal.connect(finished)

    worker = threading.Thread(target=thread.run)
    worker.start()
    while thread._process is None:  # noqa: SLF001 - regression probes thread state.
        time.sleep(0.001)

    thread.cancel()
    worker.join(timeout=1.0)
    for _ in range(10):
        qapp.processEvents()
        if finished.called:
            break
        time.sleep(0.001)

    assert worker.is_alive() is False
    assert process.terminated is True
    assert process.killed is False
    finished.assert_called_once_with(False, "Build cancelled by user")


def test_get_docker_cmd_prefers_native_docker() -> None:
    with patch.object(docker_manager.shutil, "which") as which:
        which.side_effect = lambda name: "/usr/bin/docker" if name == "docker" else None

        assert docker_manager.get_docker_cmd() == ["docker"]


def test_get_docker_cmd_uses_wsl_fallback_on_windows() -> None:
    with (
        patch.object(docker_manager.os, "name", "nt"),
        patch.object(docker_manager.shutil, "which") as which,
    ):
        which.side_effect = lambda name: (
            "C:/Windows/System32/wsl.exe" if name == "wsl" else None
        )

        assert docker_manager.get_docker_cmd() == ["wsl", "docker"]


def test_get_docker_cmd_falls_back_to_bare_docker() -> None:
    with patch.object(docker_manager.shutil, "which", return_value=None):
        assert docker_manager.get_docker_cmd() == ["docker"]


def test_docker_check_thread_emits_success_when_probe_passes() -> None:
    thread = DockerCheckThread()
    result = MagicMock()
    thread.result.connect(result)

    with patch("src.launchers.docker_manager.secure_run") as secure_run:
        thread.run()

    secure_run.assert_called_once()
    result.assert_called_once_with(True)


def test_docker_check_thread_emits_false_when_probe_rejected() -> None:
    thread = DockerCheckThread()
    result = MagicMock()
    thread.result.connect(result)

    with patch("src.launchers.docker_manager.secure_run") as secure_run:
        secure_run.side_effect = docker_manager.SecureSubprocessError("blocked")
        thread.run()

    result.assert_called_once_with(False)


@patch.object(Path, "exists", return_value=False)
def test_docker_build_rejects_missing_context(_mock_exists: MagicMock) -> None:
    context_path = Path("/missing")
    thread = DockerBuildThread(context_path=context_path)
    finished = MagicMock()
    thread.finished_signal.connect(finished)

    with patch("src.launchers.docker_manager.get_docker_cmd", return_value=["docker"]):
        thread.run()

    finished.assert_called_once_with(
        False,
        f"Invalid Docker context path: {context_path}",
    )


@patch("subprocess.Popen")
@patch.object(Path, "exists", return_value=True)
def test_docker_build_emits_success_and_streams_output(
    _mock_exists: MagicMock,
    mock_popen: MagicMock,
) -> None:
    process = CompletedProcess(returncode=0)
    mock_popen.return_value = process
    thread = DockerBuildThread(
        target_stage="standard",
        image_name="test_image",
        context_path=Path("/fake/context"),
        dockerfile_path=Path("/fake/context/Dockerfile"),
        build_args={"ARG": "value"},
    )
    finished = MagicMock()
    logs = MagicMock()
    thread.finished_signal.connect(finished)
    thread.log_signal.connect(logs)

    with patch("src.launchers.docker_manager.get_docker_cmd", return_value=["docker"]):
        thread.run()

    assert process.wait_timeouts
    assert process.stdout.closed is True
    command = mock_popen.call_args.args[0]
    assert command[:3] == ["docker", "build", "-t"]
    assert command[command.index("--build-arg") : command.index("--build-arg") + 2] == [
        "--build-arg",
        "ARG=value",
    ]
    finished.assert_called_once_with(True, "Build successful.")
    logs.assert_any_call("step 1")
    logs.assert_any_call("step 2")


@patch("subprocess.Popen")
@patch.object(Path, "exists", return_value=True)
def test_docker_build_reports_nonzero_exit(
    _mock_exists: MagicMock,
    mock_popen: MagicMock,
) -> None:
    mock_popen.return_value = CompletedProcess(returncode=17)
    thread = DockerBuildThread(context_path=Path("/fake/context"))
    finished = MagicMock()
    thread.finished_signal.connect(finished)

    thread.run()

    finished.assert_called_once_with(False, "Build failed with code 17")


def test_docker_launcher_detects_primary_image() -> None:
    launcher = DockerLauncher(Path("/repo"), image_name="current:latest")

    with patch("src.launchers.docker_manager.subprocess.run") as run:
        run.return_value.returncode = 0

        assert launcher.check_image_exists() is True


def test_docker_launcher_uses_legacy_image_when_primary_missing() -> None:
    launcher = DockerLauncher(Path("/repo"), image_name="current:latest")

    with patch("src.launchers.docker_manager.subprocess.run") as run:
        run.side_effect = [
            MagicMock(returncode=1),
            MagicMock(returncode=0),
        ]

        assert launcher.check_image_exists() is True
        assert launcher.image_name == docker_manager.LEGACY_DOCKER_IMAGE_ALIASES[0]


def test_docker_launcher_returns_false_when_image_probe_fails() -> None:
    launcher = DockerLauncher(Path("/repo"), image_name="current:latest")

    with patch("src.launchers.docker_manager.subprocess.run") as run:
        run.side_effect = OSError("docker unavailable")

        assert launcher.check_image_exists() is False


def test_docker_launcher_builds_linux_gpu_drake_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISPLAY", ":7")
    launcher = DockerLauncher(Path("/repo"), image_name="current:latest")

    with patch("src.launchers.docker_manager.get_docker_cmd", return_value=["docker"]):
        cmd = launcher.build_launch_command(
            "drake",
            Path("/repo/models/drake/app.py"),
            use_gpu=True,
        )

    assert cmd[:2] == ["docker", "run"]
    assert "--gpus=all" in cmd
    assert "7000:7000" in cmd
    assert f"{Path('/repo')}:/workspace" in cmd
    assert "-m" in cmd
    assert "src.drake_gui_app" in cmd


def test_docker_launcher_converts_windows_path_for_wsl() -> None:
    launcher = DockerLauncher(Path("C:/repo"), image_name="current:latest")

    with patch(
        "src.launchers.docker_manager.get_docker_cmd", return_value=["wsl", "docker"]
    ):
        cmd = launcher.build_launch_command(
            "custom_dashboard",
            Path("C:/repo/models/custom/app.py"),
        )

    assert "/mnt/c/repo:/workspace" in cmd


def test_docker_launcher_launches_capture_process() -> None:
    launcher = DockerLauncher(Path("/repo"), image_name="current:latest")
    process = MagicMock()

    with (
        patch.object(launcher, "build_launch_command", return_value=["docker", "run"]),
        patch(
            "src.launchers.docker_manager.subprocess.Popen", return_value=process
        ) as popen,
    ):
        assert (
            launcher.launch_container(
                "custom", "model", Path("/repo/model.py"), capture_output=True
            )
            is process
        )

    assert popen.call_args.kwargs["stdout"] is subprocess.PIPE


def test_docker_launcher_returns_none_when_launch_fails() -> None:
    launcher = DockerLauncher(Path("/repo"), image_name="current:latest")

    with (
        patch.object(launcher, "build_launch_command", return_value=["docker", "run"]),
        patch(
            "src.launchers.docker_manager.subprocess.Popen",
            side_effect=OSError("no docker"),
        ),
    ):
        assert (
            launcher.launch_container("custom", "model", Path("/repo/model.py")) is None
        )
