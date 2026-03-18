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
    is_vcxsrv_running,
    start_vcxsrv,
)


@pytest.fixture
def manager():
    with patch.object(Path, "mkdir"), patch.object(Path, "exists", return_value=False):
        return ProcessManager(repo_root=PureWindowsPath("/fake/repo"))  # type: ignore[arg-type]


def test_init_log_file_truncates_if_large():
    # If file size is > 2MB, it truncates
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_stat = MagicMock()
    mock_stat.st_size = 3 * 1024 * 1024  # 3MB
    mock_path.stat.return_value = mock_stat
    mock_path.read_text.return_value = "line1\nline2\nline3\n"

    with (
        patch(
            "src.launchers.launcher_process_manager.Path.home",
            return_value=PureWindowsPath("/fake/home"),
        ),
        patch.object(Path, "mkdir"),
        patch.object(ProcessManager, "_init_log_file", autospec=True),
    ):
        mgr = ProcessManager(repo_root=PureWindowsPath("/fake/repo"))  # type: ignore[arg-type]

        # Now test init_log_file specifically
        mgr._log_file_path = mock_path
        mgr._log_dir = MagicMock()
        # Run the actual method unmocked on our mgr
        # wait, ProcessManager._init_log_file is bound, so we have to use original

    # Let's test the logic directly:
    mgr2 = ProcessManager.__new__(ProcessManager)
    mgr2._log_dir = MagicMock()
    mgr2._log_file_path = mock_path

    # We call the real init_log_file
    ProcessManager._init_log_file(mgr2)
    mock_path.read_text.assert_called_once()
    mock_path.write_text.assert_called_once()


def test_get_subprocess_env(manager):
    env = manager.get_subprocess_env()
    repo_str = str(manager.repo_root)
    src_str = str(manager.repo_root / "src")

    assert repo_str in env["PYTHONPATH"]
    assert src_str in env["PYTHONPATH"]


@patch("src.launchers.launcher_process_manager.datetime")
@patch("builtins.open", new_callable=mock_open)
def test_write_log_line(mock_open_file, mock_datetime, manager):
    mock_datetime.datetime.now.return_value.strftime.return_value = "2023"
    manager._write_log_line("TestApp", "Hello")

    mock_open_file.assert_called_once_with(
        manager._log_file_path, "a", encoding="utf-8"
    )
    mock_open_file().write.assert_called_once_with("[2023] [TestApp] Hello\n")


def test_emit_output(manager):
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


def test_attach_process(manager):
    mock_proc = MagicMock()

    with patch("threading.Thread") as mock_thread_class:
        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread

        manager.attach_process("TestApp", mock_proc)

        assert manager.running_processes["TestApp"] == mock_proc
        assert "TestApp" in manager._output_threads
        mock_thread.start.assert_called_once()


def test_stream_output(manager):
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


@patch("subprocess.Popen")
def test_launch_script_unified(mock_popen, manager):
    manager.use_separate_terminals = False

    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc

    with patch("threading.Thread") as mock_thread:
        res = manager.launch_script(
            "Test", PureWindowsPath("/fake/script.py"), PureWindowsPath("/fake/cwd")
        )

        assert res == mock_proc
        assert manager.running_processes["Test"] == mock_proc
        mock_thread.assert_called_once()
        mock_thread().start.assert_called_once()


@patch("subprocess.Popen")
def test_launch_script_separate_term(mock_popen, manager):
    manager.use_separate_terminals = True

    script_path = PureWindowsPath("/fake/script.py")
    cwd_path = PureWindowsPath("/fake/cwd")
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        manager.launch_script("Test", script_path, cwd_path)
        mock_popen.assert_called_once()
        assert "cmd /c" in mock_popen.call_args[0][0]

    mock_popen.reset_mock()

    with patch("os.name", "posix"):
        manager.launch_script(
            "Test", PureWindowsPath("/fake/script.py"), PureWindowsPath("/fake/cwd")
        )
        mock_popen.assert_called_once()
        assert (
            "python" in mock_popen.call_args[0][0][0]
            or "sys.executable" in mock_popen.call_args[0][0][0]
            or str(mock_popen.call_args[0][0][0]).endswith("python.exe")
        )


@patch("subprocess.Popen")
def test_launch_module_unified(mock_popen, manager):
    manager.use_separate_terminals = False

    with patch("threading.Thread"):
        res = manager.launch_module("Test", "my_module", PureWindowsPath("/fake/cwd"))
        mock_popen.assert_called_once()
        assert res is not None


@patch("subprocess.Popen", side_effect=OSError("Boom"))
def test_launch_module_oserror(mock_popen, manager):
    res = manager.launch_module("Test", "my_module", PureWindowsPath("/fake/cwd"))
    assert res is None


@patch("subprocess.Popen")
def test_launch_in_wsl(mock_popen, manager):
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        res = manager.launch_in_wsl("C:\\fake\\script.py")
        assert res is True
        mock_popen.assert_called_once()


@patch("subprocess.Popen")
def test_launch_module_in_wsl(mock_popen, manager):
    with patch("os.name", "posix"):
        res = manager.launch_module_in_wsl(
            "my_module", cwd=PureWindowsPath("C:\\fake\\cwd")
        )
        assert res is True
        mock_popen.assert_called_once()


def test_convert_to_wsl_path(manager):
    assert manager._convert_to_wsl_path("C:\\path\\to\\file") == "/mnt/c/path/to/file"
    assert manager._convert_to_wsl_path("D:\\path") == "/mnt/d/path"
    assert manager._convert_to_wsl_path("/already/wsl/path") == "/already/wsl/path"


@patch("src.launchers.launcher_process_manager.kill_process_tree")
def test_cleanup_processes(mock_kill, manager):
    mock_proc1 = MagicMock()
    mock_proc1.poll.return_value = None  # Running

    mock_proc2 = MagicMock()
    mock_proc2.poll.return_value = 0  # Not running

    manager.running_processes = {"proc1": mock_proc1, "proc2": mock_proc2}

    mock_kill.return_value = True  # kill successful

    manager.cleanup_processes()

    mock_kill.assert_called_once_with(mock_proc1.pid)
    assert len(manager.running_processes) == 0


def test_is_process_running(manager):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    manager.running_processes["Test"] = mock_proc

    assert manager.is_process_running("Test") is True
    assert manager.is_process_running("Missing") is False

    mock_proc.poll.return_value = 0
    assert manager.is_process_running("Test") is False


@patch("subprocess.run")
def test_is_vcxsrv_running(mock_run):
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
def test_start_vcxsrv(mock_exists, mock_popen, mock_is_running):
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        assert start_vcxsrv() is True
        mock_popen.assert_called_once()

    with patch("os.name", "posix"):
        assert start_vcxsrv() is False


def test_init_log_file_oserror(manager):
    with patch.object(Path, "mkdir", side_effect=OSError("Boom")):
        manager._init_log_file()  # Should silently pass


def test_get_log_path():
    path = ProcessManager.get_log_path()
    assert path.name == "process_output.log"


@patch("src.launchers.launcher_process_manager.datetime")
@patch("builtins.open", side_effect=OSError("Boom"))
def test_write_log_line_oserror(mock_open_file, mock_datetime, manager):
    manager._write_log_line("TestApp", "Hello")  # Should pass


def test_stream_output_runtime_error(manager):
    mock_proc = MagicMock()
    mock_proc.stdout.readline.side_effect = RuntimeError("Boom")
    mock_proc.wait.return_value = 0
    manager._stream_output("TestApp", mock_proc)  # Should catch exception


@patch("subprocess.Popen")
def test_launch_script_keep_terminal(mock_popen, manager):
    manager.use_separate_terminals = True
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        manager.launch_script(
            "Test",
            PureWindowsPath("script.py"),
            PureWindowsPath("."),
            keep_terminal_open=True,
        )
        assert "& pause" in mock_popen.call_args[0][0]


@patch("subprocess.Popen", side_effect=OSError("Boom"))
def test_launch_script_oserror(mock_popen, manager):
    res = manager.launch_script(
        "Test", PureWindowsPath("script.py"), PureWindowsPath(".")
    )
    assert res is None


@patch("subprocess.Popen")
def test_launch_module_keep_terminal(mock_popen, manager):
    manager.use_separate_terminals = True
    with (
        patch("src.launchers.launcher_process_manager.os.name", "nt"),
        patch.dict("os.environ", {}),
    ):
        # We need to simulate the env stuff in launch_module
        manager.launch_module(
            "Test", "my_module", PureWindowsPath("."), keep_terminal_open=True
        )
        assert "& pause" in mock_popen.call_args[0][0]


@patch("subprocess.Popen")
def test_launch_in_wsl_posix(mock_popen, manager):
    with patch("os.name", "posix"):
        res = manager.launch_in_wsl("script.py")
        assert res is True
        mock_popen.assert_called_once()


@patch("subprocess.Popen", side_effect=OSError("Boom"))
def test_launch_in_wsl_oserror(mock_popen, manager):
    assert manager.launch_in_wsl("script.py") is False


@patch("subprocess.Popen")
def test_launch_module_in_wsl_no_cwd(mock_popen, manager):
    with patch("os.name", "posix"):
        res = manager.launch_module_in_wsl("my_module")
        assert res is True


@patch("subprocess.Popen")
def test_launch_module_in_wsl_nt(mock_popen, manager):
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        res = manager.launch_module_in_wsl("my_module")
        assert res is True


@patch("subprocess.Popen", side_effect=OSError("Boom"))
def test_launch_module_in_wsl_oserror(mock_popen, manager):
    assert manager.launch_module_in_wsl("my_module") is False


@patch("src.launchers.launcher_process_manager.kill_process_tree")
def test_cleanup_processes_fallback(mock_kill, manager):
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


def test_cleanup_processes_oserror(manager):
    mock_proc = MagicMock()
    mock_proc.poll.side_effect = OSError("Boom")
    manager.running_processes = {"proc": mock_proc}
    manager.cleanup_processes()  # Should not crash
    assert len(manager.running_processes) == 0


@patch("subprocess.run", side_effect=OSError("Boom"))
def test_is_vcxsrv_running_oserror(mock_run):
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        assert is_vcxsrv_running() is False


@patch("src.launchers.launcher_process_manager.is_vcxsrv_running", return_value=True)
def test_start_vcxsrv_already_running(mock_is_running):
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        assert start_vcxsrv() is True


@patch("src.launchers.launcher_process_manager.is_vcxsrv_running", return_value=False)
@patch("subprocess.Popen", side_effect=ImportError("Boom"))
@patch.object(Path, "exists", return_value=True)
def test_start_vcxsrv_import_error(mock_exists, mock_popen, mock_is_running):
    # Loop over VCXSRV_PATHS but all raise ImportError
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        assert start_vcxsrv() is False


def test_subprocess_constants_fallback():
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


def test_get_subprocess_env_already_exists(manager):
    repo_str = str(manager.repo_root)
    src_str = str(manager.repo_root / "src")

    with patch.dict("os.environ", {"PYTHONPATH": f"{repo_str}{os.pathsep}{src_str}"}):
        env = manager.get_subprocess_env()
        # Should not append them twice
        assert env["PYTHONPATH"] == f"{repo_str}{os.pathsep}{src_str}"


def test_get_subprocess_env_empty(manager):
    with patch.dict("os.environ", clear=True):
        env = manager.get_subprocess_env()
        separator = ";" if __import__("os").name == "nt" else ":"
        assert separator in env["PYTHONPATH"]


def test_stream_output_empty_lines(manager):
    mock_proc = MagicMock()
    # readline returns empty bytes to trigger immediate exit,
    # and b"   \n" to trigger empty stripped line
    mock_proc.stdout.readline.side_effect = [b"    \n", b""]
    mock_proc.stderr.readline.side_effect = [b"\n", b""]
    mock_proc.wait.return_value = 0
    manager._emit_output = MagicMock()

    manager._stream_output("TestApp", mock_proc)

    manager._emit_output.assert_called_once_with("TestApp", "[exited with code 0]")


def test_stream_output_no_streams(manager):
    mock_proc = MagicMock()
    mock_proc.stdout = None
    mock_proc.stderr = None
    mock_proc.wait.return_value = 0
    manager._emit_output = MagicMock()

    manager._stream_output("TestApp", mock_proc)

    manager._emit_output.assert_called_once_with("TestApp", "[exited with code 0]")


@patch("subprocess.Popen")
def test_launch_module_unified_nt_pythonpath(mock_popen, manager):
    manager.use_separate_terminals = False

    repo_str = str(manager.repo_root)
    src_str = str(manager.repo_root / "src")

    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        # True, True branch (neither in path, passed via explicit env)
        manager.launch_module(
            "Test", "my_module", PureWindowsPath("."), env={"PYTHONPATH": "some_other"}
        )
        env_passed = mock_popen.call_args[1]["env"]
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
        env_passed = mock_popen.call_args[1]["env"]
        # Shouldn't duplicate
        assert env_passed["PYTHONPATH"] == f"{repo_str};{src_str}"

        # False, True branch
        manager.launch_module(
            "Test",
            "my_module",
            PureWindowsPath("."),
            env={"PYTHONPATH": f"{repo_str};foo"},
        )
        env_passed = mock_popen.call_args[1]["env"]
        assert env_passed["PYTHONPATH"] == f"{src_str};{repo_str};foo"

        # True, False branch
        manager.launch_module(
            "Test",
            "my_module",
            PureWindowsPath("."),
            env={"PYTHONPATH": f"{src_str};foo"},
        )
        env_passed = mock_popen.call_args[1]["env"]
        assert env_passed["PYTHONPATH"] == f"{repo_str};{src_str};foo"

    with patch("os.name", "posix"):
        manager.launch_module("Test", "my_module", PureWindowsPath("."))
        assert mock_popen.called


@patch("subprocess.Popen")
def test_launch_module_separate_terminals(mock_popen, manager):
    manager.use_separate_terminals = True

    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        manager.launch_module(
            "Test", "my_module", PureWindowsPath("."), keep_terminal_open=False
        )
        assert "cmd /c" in mock_popen.call_args[0][0]

    with patch("os.name", "posix"):
        manager.launch_module(
            "Test", "my_module", PureWindowsPath("."), keep_terminal_open=False
        )
        assert mock_popen.call_args[0][0][1] == "-m"


@patch("src.launchers.launcher_process_manager.is_vcxsrv_running", return_value=False)
@patch.object(Path, "exists", return_value=False)
def test_start_vcxsrv_not_found(mock_exists, mock_is_running):
    with patch("src.launchers.launcher_process_manager.os.name", "nt"):
        assert start_vcxsrv() is False
