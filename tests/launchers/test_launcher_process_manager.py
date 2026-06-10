import subprocess  # noqa: E402

if not hasattr(subprocess, "CREATE_NEW_CONSOLE"):
    subprocess.CREATE_NEW_CONSOLE = 0x00000010  # type: ignore
if not hasattr(subprocess, "CREATE_NO_WINDOW"):
    subprocess.CREATE_NO_WINDOW = 0x08000000  # type: ignore  # type: ignore

"""Tests for launcher_process_manager."""

import os  # noqa: E402  # noqa: E402
from pathlib import Path, PureWindowsPath  # noqa: E402
from unittest.mock import MagicMock, mock_open, patch  # noqa: E402

import pytest  # noqa: E402
from src.launchers.launcher_process_manager import (  # noqa: E402
    ProcessManager,
    _assign_to_job,
    is_vcxsrv_running,
    start_vcxsrv,
)

# Convenience patch targets used by launch_script / launch_module tests
_SECURE_POPEN = "src.launchers.launcher_process_manager.secure_popen"
_VALIDATE_SCRIPT = "src.launchers.launcher_process_manager.validate_script_path"


@pytest.fixture
def manager() -> ProcessManager:
    with patch.object(Path, "mkdir"), patch.object(Path, "exists", return_value=False):
        return ProcessManager(repo_root=PureWindowsPath("/fake/repo"))  # type: ignore[arg-type]


def test_init_log_file_truncates_if_large() -> None:
    # If file size is > 2MB, it truncates
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_stat = MagicMock()
    mock_stat.st_size = 3 * 1024 * 1024  # 3MB
    mock_path.stat.return_value = mock_stat

    mgr2 = ProcessManager.__new__(ProcessManager)
    mgr2._log_dir = MagicMock()
    mgr2._log_file_path = mock_path

    # We use mock_open to mock file reading and writing
    m = mock_open(read_data="line1\nline2\nline3\n")
    with patch("builtins.open", m):
        ProcessManager._init_log_file(mgr2)

    # Verify we opened the path for reading and writing
    m.assert_any_call(mock_path, encoding="utf-8", errors="replace")
    m.assert_any_call(mock_path, "w", encoding="utf-8")

    # Verify the contents were written (since maxlen=500 and we had 3 lines, all 3 lines are written)
    handle = m()
    handle.writelines.assert_called_once()
    called_arg = handle.writelines.call_args[0][0]
    assert list(called_arg) == ["line1\n", "line2\n", "line3\n"]


def test_get_subprocess_env(manager) -> None:
    env = manager.get_subprocess_env()
    repo_str = str(manager.repo_root)
    src_str = str(manager.repo_root / "src")

    assert repo_str in env["PYTHONPATH"]
    assert src_str in env["PYTHONPATH"]


def test_get_subprocess_env_includes_extra_python_paths(manager):
    env = manager.get_subprocess_env((Path("/external/src"), Path("/external/src")))
    expected = str(Path("/external/src"))
    assert expected in env["PYTHONPATH"]
    assert env["PYTHONPATH"].count(expected) == 1


@patch("src.launchers.launcher_process_manager.datetime")
@patch("builtins.open", new_callable=mock_open)
def test_write_log_line(mock_open_file, mock_datetime, manager) -> None:
    mock_datetime.datetime.now.return_value.strftime.return_value = "2023"
    manager._write_log_line("TestApp", "Hello")

    mock_open_file.assert_called_once_with(
        manager._log_file_path, "a", encoding="utf-8"
    )
    mock_open_file().write.assert_called_once_with("[2023] [TestApp] Hello\n")


def test_emit_output(manager) -> None:
    manager._write_log_line = MagicMock()
    manager.output_callback = MagicMock()

    manager._emit_output("TestApp", "Hi")
    manager._write_log_line.assert_called_once_with("TestApp", "Hi")
    manager.output_callback.assert_called_once_with("TestApp", "Hi")

    # Test without callback
    manager.output_callback = None
    with patch("src.launchers.launcher_process_manager.logger.info") as mock_log:
        manager._emit_output("TestApp", "Hi")
        mock_log.assert_called_once()


def test_attach_process(manager) -> None:
    mock_proc = MagicMock()

    with patch("threading.Thread") as mock_thread_class:
        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread

        manager.attach_process("TestApp", mock_proc)

        assert manager.running_processes["TestApp"] == mock_proc
        assert "TestApp" in manager._output_threads
        mock_thread.start.assert_called_once()


def test_assign_to_job_fallback_is_noop() -> None:
    mock_proc = MagicMock()

    assert _assign_to_job(mock_proc) is None


def test_stream_output(manager) -> None:
    mock_proc = MagicMock()
    mock_proc.stdout.readline.side_effect = [b"line1\n", b"line2\n", b""]
    mock_proc.stderr.readline.side_effect = [b"err1\n", b""]
    mock_proc.wait.return_value = 0

    manager._emit_output = MagicMock()

    manager._stream_output("TestApp", mock_proc)

    # 2 stdout lines, 1 stderr line, 1 exit code line
    assert manager._emit_output.call_count == 4
    manager._emit_output.assert_any_call("TestApp", "line1")
    manager._emit_output.assert_any_call("TestApp", "STDERR: err1")
    manager._emit_output.assert_any_call("TestApp", "[exited with code 0]")


@patch(_VALIDATE_SCRIPT)
@patch(_SECURE_POPEN)
def test_launch_script_unified(mock_secure_popen, mock_validate, manager) -> None:
    manager.use_separate_terminals = False

    mock_proc = MagicMock()
    mock_secure_popen.return_value = mock_proc

    with patch("threading.Thread") as mock_thread:
        res = manager.launch_script(
            "Test", PureWindowsPath("/fake/script.py"), PureWindowsPath("/fake/cwd")
        )

        assert res == mock_proc
        assert manager.running_processes["Test"] == mock_proc
        mock_secure_popen.assert_called_once()
        mock_thread.assert_called_once()
        mock_thread().start.assert_called_once()


@patch(_SECURE_POPEN)
def test_launch_script_accepts_src_script_path(mock_secure_popen, tmp_path):
    """Launcher-managed scripts under src must pass subprocess validation."""
    script_path = tmp_path / "src" / "launchers" / "dummy_launcher.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("pass\n", encoding="utf-8")

    manager = ProcessManager(repo_root=tmp_path)
    manager.use_separate_terminals = False
    mock_secure_popen.return_value = MagicMock()

    with patch("threading.Thread"):
        process = manager.launch_script("Test", script_path, tmp_path)

    assert process is mock_secure_popen.return_value
    mock_secure_popen.assert_called_once()


@patch(_VALIDATE_SCRIPT)
@patch(_SECURE_POPEN)
def test_launch_script_unified_passes_extra_python_paths(
    mock_secure_popen, mock_validate, manager
):
    manager.use_separate_terminals = False
    mock_secure_popen.return_value = MagicMock()

    with patch("threading.Thread"):
        manager.launch_script(
            "Test",
            PureWindowsPath("/fake/script.py"),
            PureWindowsPath("/fake/cwd"),
            extra_python_paths=(Path("/external/provider/src"),),
        )

    env_passed = mock_secure_popen.call_args[1]["env"]
    assert str(Path("/external/provider/src")) in env_passed["PYTHONPATH"]


@patch(_VALIDATE_SCRIPT)
@patch("subprocess.Popen")
@patch(_SECURE_POPEN)
def test_launch_script_separate_term(
    mock_secure_popen, mock_popen, mock_validate, manager
) -> None:
    manager.use_separate_terminals = True

    # Use PosixPath so Path() construction succeeds on Linux even when
    # os.name is patched to "nt" inside the module under test.
    script_path = Path("/fake/script.py")
    cwd_path = Path("/fake/cwd")
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        # Windows separate-terminal path still uses subprocess.Popen directly
        manager.launch_script("Test", script_path, cwd_path)
        mock_popen.assert_called_once()
        assert "cmd /c" in mock_popen.call_args[0][0]

    mock_popen.reset_mock()
    mock_secure_popen.reset_mock()

    with patch("os.name", "posix"):
        # Non-Windows separate-terminal path uses secure_popen
        manager.launch_script("Test", script_path, cwd_path)
        mock_secure_popen.assert_called_once()
        cmd_arg = mock_secure_popen.call_args[0][0]
        assert (
            "python" in cmd_arg[0]
            or "sys.executable" in cmd_arg[0]
            or str(cmd_arg[0]).endswith("python.exe")
        )


@patch(_SECURE_POPEN)
def test_launch_module_unified(mock_secure_popen, manager) -> None:
    manager.use_separate_terminals = False

    with patch("threading.Thread"):
        res = manager.launch_module("Test", "my_module", PureWindowsPath("/fake/cwd"))
        mock_secure_popen.assert_called_once()
        assert res is not None


@patch(_SECURE_POPEN, side_effect=OSError("Boom"))
def test_launch_module_oserror(mock_secure_popen, manager) -> None:
    res = manager.launch_module("Test", "my_module", PureWindowsPath("/fake/cwd"))
    assert res is None


def test_launch_module_invalid_name(manager) -> None:
    """Module names with shell metacharacters must be rejected."""
    res = manager.launch_module("Test", "bad; rm -rf /", PureWindowsPath("/fake/cwd"))
    assert res is None


def test_launch_module_in_wsl_invalid_name(manager) -> None:
    """WSL module launch with shell-injectable name must be rejected."""
    res = manager.launch_module_in_wsl("bad$(whoami)")
    assert res is False


@patch("subprocess.Popen")
def test_launch_in_wsl(mock_popen, manager) -> None:
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        res = manager.launch_in_wsl("C:\\fake\\script.py")
        assert res is True
        mock_popen.assert_called_once()


@patch("subprocess.Popen")
def test_launch_module_in_wsl(mock_popen, manager) -> None:
    with patch("os.name", "posix"):
        res = manager.launch_module_in_wsl(
            "my_module", cwd=PureWindowsPath("C:\\fake\\cwd")
        )
        assert res is True
        mock_popen.assert_called_once()


def test_convert_to_wsl_path(manager) -> None:
    assert manager._convert_to_wsl_path("C:\\path\\to\\file") == "/mnt/c/path/to/file"
    assert manager._convert_to_wsl_path("D:\\path") == "/mnt/d/path"
    assert manager._convert_to_wsl_path("/already/wsl/path") == "/already/wsl/path"


@patch("src.launchers.launcher_process_manager.kill_process_tree")
def test_cleanup_processes(mock_kill, manager) -> None:
    mock_proc1 = MagicMock()
    mock_proc1.poll.return_value = None  # Running

    mock_proc2 = MagicMock()
    mock_proc2.poll.return_value = 0  # Not running

    manager.running_processes = {"proc1": mock_proc1, "proc2": mock_proc2}

    mock_kill.return_value = True  # kill successful

    manager.cleanup_processes()

    mock_kill.assert_called_once_with(mock_proc1.pid)
    assert len(manager.running_processes) == 0


def test_is_process_running(manager) -> None:
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    manager.running_processes["Test"] = mock_proc

    assert manager.is_process_running("Test") is True
    assert manager.is_process_running("Missing") is False

    mock_proc.poll.return_value = 0
    assert manager.is_process_running("Test") is False


@patch("subprocess.run")
def test_is_vcxsrv_running(mock_run) -> None:
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        mock_result = MagicMock()
        mock_result.stdout = "vcxsrv.exe"
        mock_run.return_value = mock_result
        assert is_vcxsrv_running() is True

        mock_result.stdout = "other.exe"
        assert is_vcxsrv_running() is False

    with patch("os.name", "posix"):
        assert is_vcxsrv_running() is False


@patch("src.launchers.launcher_process_manager.is_vcxsrv_running", return_value=False)
@patch("subprocess.Popen")
@patch.object(Path, "exists", return_value=True)
def test_start_vcxsrv(mock_exists, mock_popen, mock_is_running) -> None:
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        assert start_vcxsrv() is True
        mock_popen.assert_called_once()

    with patch("os.name", "posix"):
        assert start_vcxsrv() is False


def test_init_log_file_oserror(manager) -> None:
    with patch.object(Path, "mkdir", side_effect=OSError("Boom")):
        manager._init_log_file()  # Should silently pass


def test_get_log_path() -> None:
    path = ProcessManager.get_log_path()
    assert path.name == "process_output.log"


@patch("src.launchers.launcher_process_manager.datetime")
@patch("builtins.open", side_effect=OSError("Boom"))
def test_write_log_line_oserror(mock_open_file, mock_datetime, manager) -> None:
    manager._write_log_line("TestApp", "Hello")  # Should pass


def test_stream_output_runtime_error(manager) -> None:
    mock_proc = MagicMock()
    mock_proc.stdout.readline.side_effect = RuntimeError("Boom")
    mock_proc.wait.return_value = 0
    manager._stream_output("TestApp", mock_proc)  # Should catch exception


@patch(_VALIDATE_SCRIPT)
@patch("subprocess.Popen")
def test_launch_script_keep_terminal(mock_popen, mock_validate, manager) -> None:
    manager.use_separate_terminals = True
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        manager.launch_script(
            "Test",
            PureWindowsPath("script.py"),
            PureWindowsPath("."),
            keep_terminal_open=True,
        )
        assert "& pause" in mock_popen.call_args[0][0]


@patch(_VALIDATE_SCRIPT)
@patch(_SECURE_POPEN, side_effect=OSError("Boom"))
def test_launch_script_oserror(mock_secure_popen, mock_validate, manager) -> None:
    res = manager.launch_script(
        "Test", PureWindowsPath("script.py"), PureWindowsPath(".")
    )
    assert res is None


@patch("subprocess.Popen")
def test_launch_module_keep_terminal(mock_popen, manager) -> None:
    manager.use_separate_terminals = True
    with (
        patch("src.launchers.launcher_process_manager.os.name", "nt"),
        patch.dict("os.environ", {}),
    ):
        # Windows separate-terminal path still uses subprocess.Popen directly
        manager.launch_module(
            "Test", "my_module", PureWindowsPath("."), keep_terminal_open=True
        )
        assert "& pause" in mock_popen.call_args[0][0]


@patch(_VALIDATE_SCRIPT)
@patch("subprocess.Popen")
@patch(_SECURE_POPEN)
def test_launch_script_separate_term_quotes_spaces_for_cmd(
    mock_secure_popen, mock_popen, mock_validate, manager
) -> None:
    """Windows console launch must use cmd-compatible double-quote quoting.

    Regression for #6921: shlex.quote uses POSIX single quotes, which
    cmd.exe does not honor, breaking paths that contain spaces.
    """
    manager.use_separate_terminals = True
    script_path = Path("/fake/dir with spaces/script.py")
    cwd_path = Path("/fake/cwd")
    fake_exe = r"C:\Program Files\Python\python.exe"

    with (
        patch("src.launchers.launcher_process_manager.os.name", "nt"),
        patch("src.launchers.launcher_process_manager.sys.executable", fake_exe),
    ):
        manager.launch_script("Test", script_path, cwd_path)
        cmd_str = mock_popen.call_args[0][0]

    # cmd.exe quoting uses double quotes around the spaced executable path.
    assert '"C:\\Program Files\\Python\\python.exe"' in cmd_str
    # POSIX single-quote quoting must never appear.
    assert "'" not in cmd_str
    assert "cmd /c" in cmd_str


@patch("subprocess.Popen")
def test_launch_module_separate_term_quotes_spaces_for_cmd(mock_popen, manager) -> None:
    """Windows module launch must double-quote the spaced interpreter path."""
    manager.use_separate_terminals = True
    fake_exe = r"C:\Program Files\Python\python.exe"

    with (
        patch("src.launchers.launcher_process_manager.os.name", "nt"),
        patch("src.launchers.launcher_process_manager.sys.executable", fake_exe),
        patch.dict("os.environ", {}),
    ):
        manager.launch_module("Test", "my_module", PureWindowsPath("."))
        cmd_str = mock_popen.call_args[0][0]

    assert '"C:\\Program Files\\Python\\python.exe"' in cmd_str
    assert "'" not in cmd_str
    assert "-m my_module" in cmd_str


def test_build_windows_console_cmd_helper() -> None:
    """The shared cmd-builder helper quotes args and toggles pause (DRY #6922)."""
    from src.launchers.launcher_process_manager import _build_windows_console_cmd

    args = [r"C:\Program Files\Python\python.exe", r"C:\my scripts\a.py"]
    plain = _build_windows_console_cmd(args, keep_terminal_open=False)
    paused = _build_windows_console_cmd(args, keep_terminal_open=True)

    assert plain.startswith('cmd /c "')
    assert '"C:\\Program Files\\Python\\python.exe"' in plain
    assert "& pause" not in plain

    assert paused.startswith('cmd /k "')
    assert "& pause" in paused


@patch("subprocess.Popen")
def test_launch_in_wsl_posix(mock_popen, manager) -> None:
    with patch("os.name", "posix"):
        res = manager.launch_in_wsl("script.py")
        assert res is True
        mock_popen.assert_called_once()


@patch("subprocess.Popen", side_effect=OSError("Boom"))
def test_launch_in_wsl_oserror(mock_popen, manager) -> None:
    assert manager.launch_in_wsl("script.py") is False


@patch("subprocess.Popen")
def test_launch_module_in_wsl_no_cwd(mock_popen, manager) -> None:
    with patch("os.name", "posix"):
        res = manager.launch_module_in_wsl("my_module")
        assert res is True


@patch("subprocess.Popen")
def test_launch_module_in_wsl_nt(mock_popen, manager) -> None:
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        res = manager.launch_module_in_wsl("my_module")
        assert res is True


@patch("subprocess.Popen", side_effect=OSError("Boom"))
def test_launch_module_in_wsl_oserror(mock_popen, manager) -> None:
    assert manager.launch_module_in_wsl("my_module") is False


@patch("src.launchers.launcher_process_manager.kill_process_tree")
def test_cleanup_processes_fallback(mock_kill, manager) -> None:
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.wait.side_effect = __import__("subprocess").TimeoutExpired(
        cmd="", timeout=5
    )

    manager.running_processes = {"proc": mock_proc}
    mock_kill.return_value = False  # Trigger fallback

    manager.cleanup_processes()
    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()
    assert len(manager.running_processes) == 0


def test_cleanup_processes_oserror(manager) -> None:
    mock_proc = MagicMock()
    mock_proc.poll.side_effect = OSError("Boom")
    manager.running_processes = {"proc": mock_proc}
    manager.cleanup_processes()  # Should not crash
    assert len(manager.running_processes) == 0


@patch("subprocess.run", side_effect=OSError("Boom"))
def test_is_vcxsrv_running_oserror(mock_run) -> None:
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        assert is_vcxsrv_running() is False


@patch("src.launchers.launcher_process_manager.is_vcxsrv_running", return_value=True)
def test_start_vcxsrv_already_running(mock_is_running) -> None:
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        assert start_vcxsrv() is True


@patch("src.launchers.launcher_process_manager.is_vcxsrv_running", return_value=False)
@patch("subprocess.Popen", side_effect=ImportError("Boom"))
@patch.object(Path, "exists", return_value=True)
def test_start_vcxsrv_import_error(mock_exists, mock_popen, mock_is_running) -> None:
    # Loop over VCXSRV_PATHS but all raise ImportError
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        assert start_vcxsrv() is False


def test_subprocess_constants_fallback() -> None:
    import importlib
    import subprocess

    import src.launchers.launcher_process_manager as lpm

    with (
        patch("sys.platform", "win32"),
        patch("subprocess.CREATE_NO_WINDOW", side_effect=AttributeError),
        patch("subprocess.CREATE_NEW_CONSOLE", side_effect=AttributeError),
    ):
        # We temporarily remove them from subprocess if they exist
        orig_no_win = getattr(subprocess, "CREATE_NO_WINDOW", None)
        orig_new_cons = getattr(subprocess, "CREATE_NEW_CONSOLE", None)

        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            del subprocess.CREATE_NO_WINDOW
        if hasattr(subprocess, "CREATE_NEW_CONSOLE"):
            del subprocess.CREATE_NEW_CONSOLE

        importlib.reload(lpm)

        assert lpm.CREATE_NO_WINDOW == 0x08000000
        assert lpm.CREATE_NEW_CONSOLE == 0x00000010

        if orig_no_win is not None:
            subprocess.CREATE_NO_WINDOW = orig_no_win  # type: ignore
        if orig_new_cons is not None:
            subprocess.CREATE_NEW_CONSOLE = orig_new_cons  # type: ignore

    with patch("sys.platform", "linux"):
        importlib.reload(lpm)
        assert lpm.CREATE_NO_WINDOW == 0
        assert lpm.CREATE_NEW_CONSOLE == 0

    importlib.reload(lpm)


def test_get_subprocess_env_already_exists(manager) -> None:
    repo_str = str(manager.repo_root)
    src_str = str(manager.repo_root / "src")

    with (
        patch.dict("os.environ", {"PYTHONPATH": f"{repo_str}{os.pathsep}{src_str}"}),
        patch("os.path.isdir", return_value=False),
    ):
        env = manager.get_subprocess_env()
        # Should not append them twice
        assert env["PYTHONPATH"] == f"{repo_str}{os.pathsep}{src_str}"


def test_get_subprocess_env_empty(manager) -> None:
    with patch.dict("os.environ", clear=True):
        env = manager.get_subprocess_env()
        separator = ";" if __import__("os").name == "nt" else ":"
        assert separator in env["PYTHONPATH"]


def test_stream_output_empty_lines(manager) -> None:
    mock_proc = MagicMock()
    # readline returns empty bytes to trigger immediate exit,
    # and b"   \n" to trigger empty stripped line
    mock_proc.stdout.readline.side_effect = [b"    \n", b""]
    mock_proc.stderr.readline.side_effect = [b"\n", b""]
    mock_proc.wait.return_value = 0
    manager._emit_output = MagicMock()

    manager._stream_output("TestApp", mock_proc)

    manager._emit_output.assert_called_once_with("TestApp", "[exited with code 0]")


def test_stream_output_no_streams(manager) -> None:
    mock_proc = MagicMock()
    mock_proc.stdout = None
    mock_proc.stderr = None
    mock_proc.wait.return_value = 0
    manager._emit_output = MagicMock()

    manager._stream_output("TestApp", mock_proc)

    manager._emit_output.assert_called_once_with("TestApp", "[exited with code 0]")


@patch(_SECURE_POPEN)
def test_launch_module_unified_nt_pythonpath(mock_secure_popen, manager) -> None:
    manager.use_separate_terminals = False

    repo_str = str(manager.repo_root)
    src_str = str(manager.repo_root / "src")

    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        # True, True branch (neither in path, passed via explicit env)
        manager.launch_module(
            "Test", "my_module", PureWindowsPath("."), env={"PYTHONPATH": "some_other"}
        )
        env_passed = mock_secure_popen.call_args[1]["env"]
        assert repo_str in env_passed["PYTHONPATH"]
        assert src_str in env_passed["PYTHONPATH"]
        assert "some_other" in env_passed["PYTHONPATH"]

        # False, False branch
        manager.launch_module(
            "Test",
            "my_module",
            PureWindowsPath("."),
            env={"PYTHONPATH": f"{repo_str};{src_str}"},
        )
        env_passed = mock_secure_popen.call_args[1]["env"]
        # Shouldn't duplicate
        assert env_passed["PYTHONPATH"] == f"{repo_str};{src_str}"

        # False, True branch
        manager.launch_module(
            "Test",
            "my_module",
            PureWindowsPath("."),
            env={"PYTHONPATH": f"{repo_str};foo"},
        )
        env_passed = mock_secure_popen.call_args[1]["env"]
        assert env_passed["PYTHONPATH"] == f"{src_str};{repo_str};foo"

        # True, False branch
        manager.launch_module(
            "Test",
            "my_module",
            PureWindowsPath("."),
            env={"PYTHONPATH": f"{src_str};foo"},
        )
        env_passed = mock_secure_popen.call_args[1]["env"]
        assert env_passed["PYTHONPATH"] == f"{repo_str};{src_str};foo"

    with patch("os.name", "posix"):
        manager.launch_module("Test", "my_module", PureWindowsPath("."))
        assert mock_secure_popen.called


@patch(_SECURE_POPEN)
@patch("subprocess.Popen")
def test_launch_module_separate_terminals(
    mock_popen, mock_secure_popen, manager
) -> None:
    manager.use_separate_terminals = True

    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        # Windows separate-terminal path still uses subprocess.Popen directly
        manager.launch_module(
            "Test", "my_module", PureWindowsPath("."), keep_terminal_open=False
        )
        assert "cmd /c" in mock_popen.call_args[0][0]

    with patch("os.name", "posix"):
        # Non-Windows separate-terminal path uses secure_popen
        manager.launch_module(
            "Test", "my_module", PureWindowsPath("."), keep_terminal_open=False
        )
        cmd_arg = mock_secure_popen.call_args[0][0]
        assert cmd_arg[1] == "-m"


@patch("src.launchers.launcher_process_manager.is_vcxsrv_running", return_value=False)
@patch.object(Path, "exists", return_value=False)
def test_start_vcxsrv_not_found(mock_exists, mock_is_running) -> None:
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        assert start_vcxsrv() is False
