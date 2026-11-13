"""Red-first lifecycle tests for WSL/MATLAB/Docker & observer logging (#14949)."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.launchers import docker_manager, engine_status
from src.launchers.docker_manager import DockerLauncher
from src.launchers.launcher_process_manager import ProcessManager
from src.launchers.managed_process import ManagedProcess, ProcessKind, ProcessState
from src.shared.python.core.process_safety import is_pid_alive

pytestmark = pytest.mark.unit


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src" / "lifecycle_fixtures").mkdir(parents=True)
    return root


@pytest.fixture
def manager(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> ProcessManager:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return ProcessManager(repo_root=repo_root, use_separate_terminals=False)


def test_observer_errors_are_logged_not_silently_discarded(
    manager: ProcessManager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _boom_changed(_managed: object) -> None:
        raise RuntimeError("observer-changed-boom")

    def _boom_list() -> None:
        raise RuntimeError("observer-list-boom")

    manager.on_process_changed = _boom_changed
    manager.on_process_list_changed = _boom_list

    fake_proc = MagicMock()
    fake_proc.pid = 9001
    fake_proc.poll.return_value = None

    with (
        caplog.at_level(logging.ERROR),
        patch.object(ProcessManager, "_start_exit_waiter"),
        patch.object(ProcessManager, "_schedule_running_confirmation"),
        patch("src.launchers.launcher_process_manager.kill_process_tree"),
    ):
        manager._register_process("obs-key", fake_proc, kind=ProcessKind.LOCAL)
        manager.cleanup_processes()

    assert "Process observer callback failed" in caplog.text


def test_wsl_launches_are_tracked_as_managed_processes(
    manager: ProcessManager,
) -> None:
    proc1 = MagicMock(spec=subprocess.Popen)
    proc1.pid = 10101
    proc1.poll.return_value = None
    proc2 = MagicMock(spec=subprocess.Popen)
    proc2.pid = 10102
    proc2.poll.return_value = None
    proc3 = MagicMock(spec=subprocess.Popen)
    proc3.pid = 10103
    proc3.poll.return_value = None

    with (
        patch.object(ProcessManager, "_start_exit_waiter"),
        patch.object(ProcessManager, "_schedule_running_confirmation"),
        patch(
            "src.launchers.launcher_process_manager.subprocess.Popen",
            side_effect=[proc1, proc2, proc3],
        ),
    ):
        assert manager.launch_in_wsl("C:\\repo\\sim.py", name="MuJoCo") is True
        assert manager.launch_in_wsl("C:\\repo\\sim.py", name="MuJoCo") is True
        assert manager.launch_module_in_wsl("pkg.mod", name="Pinocchio") is True

    rows = manager.list_processes()
    assert len(rows) == 3
    assert all(row["kind"] == ProcessKind.WSL.value for row in rows)
    slots = {str(row["slot_name"]) for row in rows}
    assert "MuJoCo" in slots
    assert any(slot.startswith("MuJoCo (#") for slot in slots)
    assert "Pinocchio" in slots
    assert len(manager.running_processes) == 3


def test_matlab_shim_assigns_job_and_retains_duplicates(
    manager: ProcessManager,
) -> None:
    proc1 = MagicMock(spec=subprocess.Popen)
    proc1.pid = 20201
    proc1.poll.return_value = None
    proc2 = MagicMock(spec=subprocess.Popen)
    proc2.pid = 20202
    proc2.poll.return_value = None
    events: list[tuple[str, str]] = []
    manager.on_process_changed = lambda m: events.append((m.slot_name, m.kind.value))

    with (
        patch.object(ProcessManager, "_start_exit_waiter"),
        patch.object(ProcessManager, "_schedule_running_confirmation"),
        patch("src.launchers.launcher_process_manager._assign_to_job") as assign_mock,
    ):
        manager.running_processes["matlab_2d"] = proc1
        manager.running_processes["matlab_2d"] = proc2

    rows = manager.list_processes()
    assert len(rows) == 2
    assert all(row["kind"] == ProcessKind.MATLAB.value for row in rows)
    slots = {str(row["slot_name"]) for row in rows}
    assert slots == {"matlab_2d", "matlab_2d (#2)"}
    assert assign_mock.call_count == 2
    assert ("matlab_2d", "matlab") in events
    assert ("matlab_2d (#2)", "matlab") in events


def test_fast_failing_child_never_shows_running_and_surfaces_stderr_tail(
    manager: ProcessManager,
    repo_root: Path,
) -> None:
    script = repo_root / "src" / "lifecycle_fixtures" / "import_error_fast.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            sys.stderr.write("ImportError: missing_xyz\\n")
            sys.stderr.flush()
            sys.exit(1)
            """
        ).lstrip(),
        encoding="utf-8",
    )

    statuses: list[str] = []
    toasts: list[tuple[str, str]] = []

    def _set_status(text: str, _style: str) -> None:
        statuses.append(text)

    def _show_toast(msg: str, level: str) -> None:
        toasts.append((msg, level))

    def _on_changed(managed: ManagedProcess) -> None:
        engine_status.apply_managed_process_transition(
            managed,
            _set_status,
            _show_toast,
            lambda: statuses.append("Ready"),
            success_style="ok",
            error_style="err",
        )

    manager.on_process_changed = _on_changed
    proc = manager.launch_script("TestModel", script, repo_root)
    assert proc is not None

    managed = manager.get_managed_process("TestModel")
    assert managed is not None
    engine_status.apply_post_launch_status(
        managed,
        "TestModel",
        "",
        _set_status,
        _show_toast,
        success_style="ok",
        error_style="err",
        toast_label="TestModel Launched",
    )

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        rows = manager.list_processes()
        if rows and rows[0]["state"] == ProcessState.CRASHED.value:
            break
        time.sleep(0.05)

    assert any("Starting" in s or "Crashed" in s for s in statuses)
    assert not any("Running" in s for s in statuses)
    assert any("Crashed (exit 1)" in s for s in statuses)
    assert any(level == "error" and "missing_xyz" in msg for msg, level in toasts)


def test_docker_container_naming_pid_label_port_avoidance_cleanup_and_sweep(
    repo_root: Path,
) -> None:
    launcher = DockerLauncher(repo_root, image_name="current:latest")
    cmd = launcher.build_launch_command(
        "drake", repo_root / "models" / "drake" / "app.py"
    )

    assert "--name" in cmd
    name_idx = cmd.index("--name")
    assert cmd[name_idx + 1].startswith("ud-")
    assert docker_manager.CONTAINER_LABEL_MANAGED in cmd
    assert f"{docker_manager.CONTAINER_LABEL_PID_KEY}={os.getpid()}" in cmd

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    bound_port = sock.getsockname()[1]
    sock.listen(1)
    try:
        chosen = launcher._select_host_port(preferred=bound_port, max_probes=10)
        assert chosen != bound_port
    finally:
        sock.close()

    launched_cmds: list[list[str]] = []

    def _fake_popen(popen_cmd: list[str], **_kwargs: object) -> MagicMock:
        launched_cmds.append(list(popen_cmd))
        proc = MagicMock()
        proc.pid = 30000 + len(launched_cmds)
        proc.poll.return_value = None
        return proc

    with (
        patch.object(launcher, "_select_host_port", side_effect=[7000, 7001]),
        patch("src.launchers.docker_manager.subprocess.Popen", side_effect=_fake_popen),
    ):
        p1 = launcher.launch_container(
            "drake", "Drake One", repo_root / "models" / "drake" / "a.py"
        )
        p2 = launcher.launch_container(
            "drake", "Drake Two", repo_root / "models" / "drake" / "b.py"
        )

    assert p1 is not None and p2 is not None
    assert len(launched_cmds) == 2
    c1_name = launched_cmds[0][launched_cmds[0].index("--name") + 1]
    c2_name = launched_cmds[1][launched_cmds[1].index("--name") + 1]
    assert c1_name != c2_name
    assert "7000:7000" in launched_cmds[0]
    assert "7001:7000" in launched_cmds[1]

    pm = ProcessManager(repo_root=repo_root, use_separate_terminals=False)
    pm.container_cleanup_callback = launcher.stop_all_containers
    with patch("src.launchers.docker_manager.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        pm.cleanup_processes()

    stop_cmds = [call.args[0] for call in mock_run.call_args_list]
    assert any("stop" in c and c1_name in c for c in stop_cmds)
    assert any("stop" in c and c2_name in c for c in stop_cmds)

    ps_stdout = "ud-orphan-1111\t99999\nud-live-2222\t88888\n"
    with patch("src.launchers.docker_manager.subprocess.run") as mock_run:
        mock_run.side_effect = [
            SimpleNamespace(returncode=0, stdout=ps_stdout, stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
        stopped = launcher.sweep_orphan_containers(
            pid_alive_fn=lambda pid: pid == 88888
        )

    assert stopped == ["ud-orphan-1111"]
    sweep_cmds = [call.args[0] for call in mock_run.call_args_list]
    assert any("stop" in c and "ud-orphan-1111" in c for c in sweep_cmds)
    assert not any("ud-live-2222" in c for c in sweep_cmds)


def test_launch_in_wsl_uses_pure_windows_stem_without_concrete_path(
    manager: ProcessManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = MagicMock()
    fake_proc.pid = 4321
    fake_proc.poll.return_value = None

    monkeypatch.setattr("src.launchers.launcher_process_manager.os.name", "nt")
    monkeypatch.setattr(
        "src.launchers.launcher_process_manager.Path",
        MagicMock(
            side_effect=NotImplementedError(
                "cannot instantiate 'WindowsPath' on your system"
            )
        ),
    )
    monkeypatch.setattr(
        "src.launchers.launcher_process_manager._assign_to_job",
        lambda _proc: None,
    )
    with patch("subprocess.Popen", return_value=fake_proc):
        assert manager.launch_in_wsl("C:\\fake\\script.py") is True

    slots = [
        row for row in manager.list_processes() if row["kind"] == ProcessKind.WSL.value
    ]
    assert [row["key"] for row in slots] == ["script"]


def test_is_pid_alive_never_calls_os_kill_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.shared.python.core import process_safety

    with patch.object(process_safety.os, "kill") as mock_kill:
        assert is_pid_alive(0) is False
        assert is_pid_alive(-1) is False
        with pytest.raises(TypeError, match="int"):
            is_pid_alive(True)  # type: ignore[arg-type]

        monkeypatch.setattr(process_safety.os, "name", "nt")
        if sys.platform == "win32":
            assert is_pid_alive(os.getpid()) is True
        else:
            monkeypatch.setattr(
                process_safety, "_is_pid_alive_windows", lambda pid: pid == 4242
            )
            assert is_pid_alive(4242) is True
            assert is_pid_alive(9999) is False
        mock_kill.assert_not_called()


def test_is_pid_alive_windows_exit_code_and_posix_kill_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes
    from src.shared.python.core import process_safety

    closed_handles: list[int] = []

    class _FakeKernel32:
        def __init__(self, handle: int, exit_code: int) -> None:
            self._handle = handle
            self._exit_code = exit_code

        def OpenProcess(self, _access: int, _inherit: int, _pid: int) -> int:
            return self._handle

        def GetExitCodeProcess(self, _handle: int, out_ptr: object) -> int:
            ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_ulong))[0] = self._exit_code  # type: ignore[arg-type]
            return 1

        def CloseHandle(self, handle: int) -> int:
            closed_handles.append(handle)
            return 1

    assert (
        process_safety._is_pid_alive_windows(
            111, kernel32=_FakeKernel32(0, 0), get_last_error=lambda: 5
        )
        is True
    )
    assert (
        process_safety._is_pid_alive_windows(
            111, kernel32=_FakeKernel32(0, 0), get_last_error=lambda: 87
        )
        is False
    )
    assert (
        process_safety._is_pid_alive_windows(
            111, kernel32=_FakeKernel32(0x100, 259), get_last_error=lambda: 0
        )
        is True
    )
    assert (
        process_safety._is_pid_alive_windows(
            111, kernel32=_FakeKernel32(0x200, 0), get_last_error=lambda: 0
        )
        is False
    )
    assert closed_handles == [0x100, 0x200]

    monkeypatch.setattr(process_safety.os, "name", "posix")
    with patch.object(process_safety.os, "kill", return_value=None) as mock_kill:
        assert is_pid_alive(12345) is True
        mock_kill.assert_called_once_with(12345, 0)

    with patch.object(process_safety.os, "kill", side_effect=ProcessLookupError):
        assert is_pid_alive(12345) is False

    with patch.object(process_safety.os, "kill", side_effect=PermissionError):
        assert is_pid_alive(12345) is True


def test_resolve_matlab_r2025b_executable_avoids_path_and_rejects_r2026a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.launchers import launcher_simulation

    monkeypatch.delenv("MATLAB_R2025B_EXE", raising=False)
    monkeypatch.setattr(launcher_simulation.os, "name", "nt")
    assert (
        launcher_simulation.resolve_matlab_r2025b_executable()
        == "C:/Program Files/MATLAB/R2025b/bin/matlab.exe"
    )
    monkeypatch.setattr(launcher_simulation.os, "name", "posix")
    assert (
        launcher_simulation.resolve_matlab_r2025b_executable()
        == "/usr/local/MATLAB/R2025b/bin/matlab"
    )

    monkeypatch.setenv(
        "MATLAB_R2025B_EXE", "C:/Program Files/MATLAB/R2026a/bin/matlab.exe"
    )
    with pytest.raises(ValueError, match="R2025b"):
        launcher_simulation.resolve_matlab_r2025b_executable()

    monkeypatch.setenv("MATLAB_R2025B_EXE", "matlab")
    with pytest.raises(ValueError, match="R2025b"):
        launcher_simulation.resolve_matlab_r2025b_executable()


def test_launch_matlab_app_uses_explicit_r2025b_executable_on_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.launchers import launcher_simulation

    monkeypatch.delenv("MATLAB_R2025B_EXE", raising=False)
    monkeypatch.setattr(launcher_simulation.os, "name", "nt")
    slx_file = tmp_path / "GolfSwing3D_Kinetic.slx"
    slx_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        launcher_simulation,
        "resolve_model_artifact_path",
        lambda _app, _root: slx_file,
    )
    monkeypatch.setattr(
        launcher_simulation,
        "get_model_source_root",
        lambda _app, _root: tmp_path,
    )

    host = SimpleNamespace(
        running_processes={},
        show_toast=MagicMock(),
    )
    sim = launcher_simulation.SimulationManager(host)
    app = SimpleNamespace(
        id="matlab_3d",
        name="Simscape 3D",
        path=str(slx_file),
    )
    fake_proc = MagicMock(spec=subprocess.Popen)
    with patch.object(
        launcher_simulation, "secure_popen", return_value=fake_proc
    ) as mock_popen:
        ok = sim._launch_matlab_app(app)

    assert ok is True
    launched_cmd = mock_popen.call_args.args[0]
    assert launched_cmd[0] == "C:/Program Files/MATLAB/R2025b/bin/matlab.exe"
    assert host.running_processes["matlab_3d"] is fake_proc
