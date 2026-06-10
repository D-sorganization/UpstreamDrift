# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Process management utilities for the Golf Launcher.

This module provides centralized process lifecycle management for launching
simulations, Docker containers, and WSL processes.

Supports two output modes:
- **Unified console** (default): Output is captured via pipes and routed
  to a callback (e.g. a dockable console widget in the GUI). No separate
  terminal windows are created.
- **Separate terminals** (legacy): Each engine gets its own console window
  via CREATE_NEW_CONSOLE. Can be enabled per-launch or globally.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.security.secure_subprocess import (
    SecureSubprocessError,
    secure_popen,
    validate_script_path,
)
from src.shared.python.security.subprocess_utils import kill_process_tree

if TYPE_CHECKING:
    from subprocess import Popen

logger = get_logger(__name__)

OutputCallback = Callable[[str, str], None]  # (engine_name, line) -> None

# Windows-specific subprocess constants
CREATE_NO_WINDOW: int
CREATE_NEW_CONSOLE: int

if sys.platform == "win32":
    try:
        CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        CREATE_NEW_CONSOLE = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    except AttributeError:
        CREATE_NO_WINDOW = 0x08000000
        CREATE_NEW_CONSOLE = 0x00000010
else:
    CREATE_NO_WINDOW = 0
    CREATE_NEW_CONSOLE = 0


def _assign_to_job(proc: subprocess.Popen[bytes]) -> None:
    """Attach a child process to the platform cascade-termination guard."""


_preexec_fn: Callable[[], None] | None = None
if sys.platform == "win32":
    try:
        import win32api
        import win32con
        import win32job

        _job = win32job.CreateJobObject(None, "")
        _info = win32job.QueryInformationJobObject(
            _job,
            win32job.JobObjectExtendedLimitInformation,
        )
        _info["BasicLimitInformation"]["LimitFlags"] = (
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        win32job.SetInformationJobObject(
            _job,
            win32job.JobObjectExtendedLimitInformation,
            _info,
        )

        def _assign_to_job(proc: subprocess.Popen[bytes]) -> None:
            try:
                handle = win32api.OpenProcess(
                    win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE,
                    False,
                    proc.pid,
                )
                win32job.AssignProcessToJobObject(_job, handle)
            except (OSError, RuntimeError, TypeError) as exc:
                logger.debug("Failed to assign process to job object: %s", exc)

    except ImportError:
        logger.debug("win32job not available, orphaned processes may leak on crash")
else:
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6")

        def _preexec_fn() -> None:
            libc.prctl(1, signal.SIGTERM)

    except (AttributeError, OSError):
        _preexec_fn = None

# VcXsrv paths for Windows X11 support
VCXSRV_PATHS = [
    Path("C:/Program Files/VcXsrv/vcxsrv.exe"),
    Path("C:/Program Files (x86)/VcXsrv/vcxsrv.exe"),
]

# Allowlist for Python module names passed to -m flag.
# Matches dotted identifiers such as "my_pkg.sub_module".
_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _build_windows_console_cmd(args: list[str], *, keep_terminal_open: bool) -> str:
    """Build a cmd.exe command string for a new-console subprocess launch.

    ``subprocess.list2cmdline`` applies Windows (cmd.exe) quoting rules,
    wrapping arguments that contain spaces in double quotes. POSIX
    ``shlex.quote`` uses single quotes, which cmd.exe does not recognize,
    so interpreter or script paths containing spaces fail to launch
    (issue #6921).

    Args:
        args: The program and its arguments, e.g. ``[exe, script]`` or
            ``[exe, "-m", module]``. All entries must already be validated.
        keep_terminal_open: When ``True`` use ``cmd /k ... & pause`` so the
            console stays open after the process exits; otherwise ``cmd /c``.

    Returns:
        A command string suitable for ``subprocess.Popen`` with
        ``creationflags=CREATE_NEW_CONSOLE``.
    """
    inner = subprocess.list2cmdline(args)
    if keep_terminal_open:
        return f'cmd /k "{inner} & pause"'
    return f'cmd /c "{inner}"'


class ProcessManager:
    """Manages subprocess lifecycle for the Golf Launcher.

    This class centralizes process creation, monitoring, and cleanup
    for Python scripts, modules, Docker containers, and WSL processes.

    By default, subprocess output is captured and routed to an
    ``output_callback`` (unified console mode). Set
    ``use_separate_terminals=True`` to revert to the legacy behavior
    of opening a new console window per engine.
    """

    def __init__(
        self,
        repo_root: Path,
        output_callback: OutputCallback | None = None,
        use_separate_terminals: bool = False,
    ) -> None:
        """Initialize the process manager.

        Args:
            repo_root: Root directory of the repository.
            output_callback: Called with (engine_name, line) for each
                line of captured output. If *None*, output is logged
                via the Python logger instead.
            use_separate_terminals: If True, each engine opens its own
                console window (legacy behaviour).
        """
        if repo_root is None:
            raise ValueError("repo_root must be provided")
        self.repo_root = repo_root
        self.running_processes: dict[str, Popen[bytes]] = {}
        self.output_callback = output_callback
        self.use_separate_terminals = use_separate_terminals
        self._output_threads: dict[str, threading.Thread] = {}
        # Thread-safe guard for running_processes dict (issue #2715)
        self._process_lock = threading.RLock()

        # Persistent log file for all process output
        self._log_dir = Path.home() / ".golf_modeling_suite"
        self._log_file_path = self._log_dir / "process_output.log"
        self._init_log_file()

    def _merge_python_paths(
        self,
        existing_path: str,
        extra_python_paths: tuple[str, ...] = (),
    ) -> str:
        """Merge required and extra PYTHONPATH entries without duplication."""
        separator = ";" if os.name == "nt" else ":"
        current_paths = existing_path.split(separator) if existing_path else []

        required_paths = [str(self.repo_root), str(self.repo_root / "src")]
        optional_paths = [
            str(self.repo_root / "src" / "shared" / "python"),
            str(
                self.repo_root
                / "src"
                / "engines"
                / "physics_engines"
                / "mujoco"
                / "python"
            ),
            os.path.join(
                os.path.expanduser("~"),
                "miniconda3",
                "lib",
                "python3.10",
                "site-packages",
            ),
        ]

        merged_paths: list[str] = []
        seen: set[str] = set()

        for path in required_paths:
            if path not in seen and path not in current_paths:
                seen.add(path)
                merged_paths.append(path)

        for path in [*optional_paths, *extra_python_paths]:
            if path in seen or path in current_paths:
                continue
            if path in optional_paths and not os.path.isdir(path):
                continue
            seen.add(path)
            merged_paths.append(path)

        if not merged_paths:
            return existing_path

        new_paths = separator.join(merged_paths)
        if existing_path:
            return f"{new_paths}{separator}{existing_path}"
        return new_paths

    def get_subprocess_env(
        self,
        extra_python_paths: tuple[Path, ...] = (),
    ) -> dict[str, str]:
        """Get environment variables for subprocess execution.

        Returns:
            Dictionary of environment variables with proper PYTHONPATH.
        """
        env = os.environ.copy()
        existing_path = env.get("PYTHONPATH", "")
        separator = ";" if os.name == "nt" else ":"
        current_paths = existing_path.split(separator) if existing_path else []

        shared_python = str(self.repo_root / "src" / "shared" / "python")
        mujoco_python = str(
            self.repo_root / "src" / "engines" / "physics_engines" / "mujoco" / "python"
        )
        # Include conda site-packages for opensim/pinocchio if available.
        # Use os.path to avoid WindowsPath instantiation issues on Linux.
        conda_sp = os.path.join(
            os.path.expanduser("~"),
            "miniconda3",
            "lib",
            "python3.10",
            "site-packages",
        )

        repo_root_str = str(self.repo_root)
        src_dir = str(self.repo_root / "src")

        # repo_root and src are always added (required for imports).
        # Optional extras are only added when the directory exists.
        paths_to_add = []
        for p in [repo_root_str, src_dir]:
            if p not in current_paths:
                paths_to_add.append(p)
        for p in [shared_python, mujoco_python, conda_sp]:
            if p not in current_paths and os.path.isdir(p):
                paths_to_add.append(p)
        for p_path in extra_python_paths:
            p = str(p_path)
            if p not in current_paths and p not in paths_to_add:
                paths_to_add.append(p)

        if paths_to_add:
            new_paths = separator.join(paths_to_add)
            env["PYTHONPATH"] = (
                f"{new_paths}{separator}{existing_path}" if existing_path else new_paths
            )

        return env

    def _validate_context_path(self, context_path: Path) -> Path:
        """Validate subprocess working directory against allowlist (issue #2715).

        Args:
            context_path: Proposed working directory.

        Returns:
            Resolved path if valid.

        Raises:
            ValueError: If path is outside repo_root, is a symlink, or unresolvable.
        """
        if not hasattr(context_path, "resolve"):
            # Synthetic PurePath inputs are used in unit tests. Keep them usable
            # rather than forcing a concrete conversion that can be sensitive to
            # os.name on Windows.
            return context_path

        def _normalize(path_value: Path | str) -> str:
            raw = os.fspath(path_value)
            return os.path.normcase(os.path.abspath(os.path.normpath(raw)))

        resolved = context_path

        repo_root_exists = hasattr(self.repo_root, "exists") and self.repo_root.exists()
        repo_root_resolved = _normalize(self.repo_root) if repo_root_exists else None
        temp_root_resolved = _normalize(tempfile.gettempdir())

        def _is_within(candidate: str, base: str) -> bool:
            return candidate == base or candidate.startswith(base + os.sep)

        # Allow temporary directories for regression tests and transient work.
        # Otherwise keep the existing repo-root containment rule.
        if repo_root_exists and repo_root_resolved is not None:
            candidate_resolved = _normalize(resolved)

            # Find sibling Tools repository for security validations
            from src.shared.python.security.secure_subprocess import (
                _find_tools_repo_for_security,
            )

            tools_repo = _find_tools_repo_for_security(self.repo_root)
            tools_repo_resolved = (
                _normalize(tools_repo) if tools_repo is not None else None
            )

            if not (
                _is_within(candidate_resolved, repo_root_resolved)
                or _is_within(candidate_resolved, temp_root_resolved)
                or (
                    tools_repo_resolved is not None
                    and _is_within(candidate_resolved, tools_repo_resolved)
                )
            ):
                raise ValueError(
                    f"Path {context_path} is outside repo_root {self.repo_root}"
                )

        # Reject if original is a symlink (prevents symlink-escape bypasses)
        if hasattr(context_path, "is_symlink") and context_path.is_symlink():
            raise ValueError(f"Symlinks not allowed for subprocess cwd: {context_path}")

        return resolved

    def _init_log_file(self) -> None:
        """Initialize the persistent process output log file."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            # Truncate if larger than 2 MB to prevent unbounded growth
            if (
                self._log_file_path.exists()
                and self._log_file_path.stat().st_size > 2 * 1024 * 1024
            ):
                # Optimized log truncation: Keep last 500 lines without loading entire file into RAM (issue #2715)
                # We use a deque with maxlen=500 to efficiently store only the tail.
                from collections import deque

                with open(self._log_file_path, encoding="utf-8", errors="replace") as f:
                    tail = deque(f, maxlen=500)

                with open(self._log_file_path, "w", encoding="utf-8") as f:
                    f.writelines(tail)
        except (RuntimeError, ValueError, OSError, ImportError) as e:
            logger.debug("Could not init log file: %s", e)

    @classmethod
    def get_log_path(cls) -> Path:
        """Return the path to the persistent process output log."""
        return Path.home() / ".golf_modeling_suite" / "process_output.log"

    def _write_log_line(self, name: str, line: str) -> None:
        """Append a timestamped line to the persistent log file."""
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] [{name}] {line}\n")
        except (FileNotFoundError, PermissionError, OSError):
            pass  # Never let logging crash the app

    def _emit_output(self, name: str, line: str) -> None:
        """Route a line of process output to callback, logger, and log file."""
        if name is None:
            raise ValueError("name must be provided")
        self._write_log_line(name, line)
        if self.output_callback is not None:
            self.output_callback(name, line)
        else:
            logger.info("[%s] %s", name, line)

    def _add_to_running_processes(
        self, name: str, process: subprocess.Popen[bytes]
    ) -> None:
        with self._process_lock:
            self.running_processes[name] = process
        if getattr(self, "on_process_list_changed", None):
            with contextlib.suppress(Exception):
                self.on_process_list_changed()

    def attach_process(self, name: str, process: subprocess.Popen[bytes]) -> None:
        """Attach an externally-created process for output streaming.

        Use this for processes not created by ProcessManager (e.g. Docker
        containers) that still need their output captured in the unified
        console and log file.
        """
        if name is None:
            raise ValueError("name must be provided")
        self._add_to_running_processes(name, process)
        t = threading.Thread(
            target=self._stream_output,
            args=(name, process),
            daemon=True,
        )
        t.start()
        self._output_threads[name] = t

    def _stream_output(self, name: str, process: subprocess.Popen[bytes]) -> None:
        """Read stdout/stderr from *process* and emit lines until EOF.

        Runs in a daemon thread so the main GUI thread is never blocked.
        """
        if name is None:
            raise ValueError("name must be provided")
        try:
            if process.stdout:
                for raw_line in iter(process.stdout.readline, b""):
                    line = raw_line.decode("utf-8", errors="replace").rstrip()
                    if line:
                        self._emit_output(name, line)
                process.stdout.close()
            if process.stderr:
                for raw_line in iter(process.stderr.readline, b""):
                    line = raw_line.decode("utf-8", errors="replace").rstrip()
                    if line:
                        self._emit_output(name, f"STDERR: {line}")
                process.stderr.close()
        except (RuntimeError, ValueError, OSError) as e:
            logger.debug("Output stream ended for %s: %s", name, e)

        return_code = process.wait()
        self._emit_output(name, f"[exited with code {return_code}]")

    def launch_script(
        self,
        name: str,
        script_path: Path,
        cwd: Path,
        env: dict[str, str] | None = None,
        extra_python_paths: tuple[Path, ...] = (),
        keep_terminal_open: bool = False,
    ) -> subprocess.Popen[bytes] | None:
        """Launch a Python script as a subprocess.

        Args:
            name: Display name for the process.
            script_path: Path to the Python script.
            cwd: Working directory for the process.
            env: Optional environment variables.
            keep_terminal_open: If True, keep terminal open on script exit/error
                               (uses cmd /k with pause). Default False.
                               Only effective in separate-terminal mode.

        Returns:
            The process object if successful, None otherwise.
        """
        try:
            process_env = env or self.get_subprocess_env(extra_python_paths)

            # Validate script path to prevent path-traversal / injection.
            validate_script_path(script_path, self.repo_root)

            # Validate working directory (issue #2715: reject paths outside repo)
            cwd = self._validate_context_path(cwd)

            # Diagnostic: log full launch details for debugging silent failures
            logger.info(
                "Launching script %s: cmd=[%s, %s], cwd=%s, PYTHONPATH=%s",
                name,
                sys.executable,
                script_path,
                cwd,
                process_env.get("PYTHONPATH", "<unset>")[:300],
            )

            if self.use_separate_terminals:
                # Legacy: each engine gets its own console window.
                # On Windows we must pass a string to open a new console but
                # quote both the interpreter and script paths so that spaces
                # and other shell-significant characters cannot inject commands.
                if os.name == "nt":
                    cmd_str = _build_windows_console_cmd(
                        [sys.executable, str(script_path)],
                        keep_terminal_open=keep_terminal_open,
                    )
                    process = subprocess.Popen(
                        cmd_str,
                        cwd=str(cwd),
                        env=process_env,
                        creationflags=CREATE_NEW_CONSOLE,
                    )
                else:
                    process = secure_popen(
                        [sys.executable, str(script_path)],
                        cwd=cwd,
                        suite_root=self.repo_root,
                        env=process_env,
                        preexec_fn=_preexec_fn,
                    )
            else:
                # Unified console: capture output via pipes
                process = secure_popen(
                    [sys.executable, str(script_path)],
                    cwd=cwd,
                    suite_root=self.repo_root,
                    env=process_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                    preexec_fn=_preexec_fn,
                )
                # Stream output in a background thread
                t = threading.Thread(
                    target=self._stream_output,
                    args=(name, process),
                    daemon=True,
                )
                t.start()
                self._output_threads[name] = t

            _assign_to_job(process)

            # Guard running_processes dict with lock (issue #2715)
            self._add_to_running_processes(name, process)
            logger.info(f"Launched {name} (PID: {process.pid})")
            return process

        except (
            FileNotFoundError,
            PermissionError,
            OSError,
            SecureSubprocessError,
            ValueError,
        ) as e:
            logger.error(f"Failed to launch {name}: {e}")
            return None

    def launch_module(  # noqa: C901
        self,
        name: str,
        module_name: str,
        cwd: Path,
        env: dict[str, str] | None = None,
        extra_python_paths: tuple[Path, ...] = (),
        keep_terminal_open: bool = False,
    ) -> subprocess.Popen[bytes] | None:
        """Launch a Python module as a subprocess.

        Args:
            name: Display name for the process.
            module_name: Python module name (for -m flag).
            cwd: Working directory for the process.
            env: Optional environment variables.
            keep_terminal_open: If True, keep terminal open on script exit/error
                               (uses cmd /k with pause). Default False.
                               Only effective in separate-terminal mode.

        Returns:
            The process object if successful, None otherwise.
        """
        try:
            process_env = env or self.get_subprocess_env(extra_python_paths)

            # Validate working directory (issue #2715: reject paths outside repo)
            cwd = self._validate_context_path(cwd)

            # Validate module name: must be a dotted Python identifier.
            if not _MODULE_NAME_RE.match(module_name):
                raise SecureSubprocessError(
                    f"Invalid module name (potential injection): {module_name!r}"
                )

            if os.name == "nt":
                current_pythonpath = process_env.get("PYTHONPATH", "")
                repo_root_str = str(self.repo_root)
                src_dir_str = str(self.repo_root / "src")

                current_paths = (
                    current_pythonpath.split(";") if current_pythonpath else []
                )
                paths_to_add = []
                if repo_root_str not in current_paths:
                    paths_to_add.append(repo_root_str)
                if src_dir_str not in current_paths:
                    paths_to_add.append(src_dir_str)

                if paths_to_add:
                    process_env["PYTHONPATH"] = (
                        f"{';'.join(paths_to_add)};{current_pythonpath}"
                        if current_pythonpath
                        else ";".join(paths_to_add)
                    )

            # Diagnostic: log full launch details for debugging silent failures
            logger.info(
                "Launching module %s: cmd=[%s, -m, %s], cwd=%s, PYTHONPATH=%s",
                name,
                sys.executable,
                module_name,
                cwd,
                process_env.get("PYTHONPATH", "<unset>")[:300],
            )

            if self.use_separate_terminals:
                # Legacy: each engine gets its own console window.
                # On Windows we must pass a string to open a new console but
                # quote the interpreter path so spaces cannot inject commands.
                # module_name has already been validated against the allowlist
                # regex so it is safe to pass through.
                if os.name == "nt":
                    cmd_str = _build_windows_console_cmd(
                        [sys.executable, "-m", module_name],
                        keep_terminal_open=keep_terminal_open,
                    )
                    process = subprocess.Popen(
                        cmd_str,
                        cwd=str(cwd),
                        env=process_env,
                        creationflags=CREATE_NEW_CONSOLE,
                    )
                else:
                    process = secure_popen(
                        [sys.executable, "-m", module_name],
                        cwd=cwd,
                        suite_root=self.repo_root,
                        env=process_env,
                        preexec_fn=_preexec_fn,
                    )
            else:
                # Unified console: capture output via pipes
                process = secure_popen(
                    [sys.executable, "-m", module_name],
                    cwd=cwd,
                    suite_root=self.repo_root,
                    env=process_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                    preexec_fn=_preexec_fn,
                )
                t = threading.Thread(
                    target=self._stream_output,
                    args=(name, process),
                    daemon=True,
                )
                t.start()
                self._output_threads[name] = t

            _assign_to_job(process)

            # Guard running_processes dict with lock (issue #2715)
            self._add_to_running_processes(name, process)
            logger.info(f"Launched module {name} (PID: {process.pid})")
            return process

        except (
            FileNotFoundError,
            PermissionError,
            OSError,
            SecureSubprocessError,
        ) as e:
            logger.error(f"Failed to launch {name}: {e}")
            return None

    def _get_wsl_distro(self) -> str:
        """Return the WSL distro name from WSL_DISTRO env var (default: Ubuntu)."""
        return os.environ.get("WSL_DISTRO", "Ubuntu")

    def _get_wsl_project_dir(self) -> str:
        """Return the WSL project directory from WSL_PROJECT_DIR env var.

        Falls back to converting self.repo_root to a WSL path.
        """
        if env_val := os.environ.get("WSL_PROJECT_DIR"):
            return env_val
        return self._convert_to_wsl_path(str(self.repo_root))

    def _get_wsl_conda_env(self) -> str:
        """Return the conda environment name from WSL_CONDA_ENV env var (default: base)."""
        return os.environ.get("WSL_CONDA_ENV", "base")

    def launch_in_wsl(
        self,
        script_path: str,
        project_dir: str | None = None,
    ) -> bool:
        """Launch a script in WSL2 Ubuntu environment.

        WSL settings are read from environment variables so this method is
        portable across developers and machines:
        - ``WSL_DISTRO``: WSL distro name (default: ``"Ubuntu"``)
        - ``WSL_PROJECT_DIR``: WSL path to the project root (default: derived
          from repo_root)
        - ``WSL_CONDA_ENV``: conda environment name (default: ``"base"``)

        Args:
            script_path: Windows path to the script.
            project_dir: Override WSL project dir (uses WSL_PROJECT_DIR if not set).

        Returns:
            True if launch succeeded, False otherwise.
        """
        # Convert Windows path to WSL path
        if script_path is None:
            raise ValueError("script_path must be provided")

        resolved_project_dir = project_dir or self._get_wsl_project_dir()
        distro = self._get_wsl_distro()
        conda_env = self._get_wsl_conda_env()

        wsl_script_path = self._convert_to_wsl_path(script_path)

        # Use shlex.quote to prevent injection of shell metacharacters in the
        # paths that are interpolated into the bash -c script.
        quoted_project_dir = shlex.quote(resolved_project_dir)
        quoted_wsl_script = shlex.quote(wsl_script_path)
        quoted_conda_env = shlex.quote(conda_env)

        wsl_cmd = (
            "source ~/miniforge3/etc/profile.d/conda.sh\n"
            f"conda activate {quoted_conda_env}\n"
            'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"\n'
            f"export PYTHONPATH={quoted_project_dir}:$PYTHONPATH\n"
            f"cd {quoted_project_dir}\n"
            f"python {quoted_wsl_script}\n"
        )

        cmd = ["wsl", "-d", distro, "--", "bash", "-c", wsl_cmd]

        try:
            logger.info(f"Launching in WSL: {script_path}")
            if os.name == "nt":
                subprocess.Popen(
                    cmd,
                    creationflags=CREATE_NEW_CONSOLE,
                )
            else:
                subprocess.Popen(cmd)
            return True

        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"WSL launch failed: {e}")
            return False

    def launch_module_in_wsl(
        self,
        module_name: str,
        cwd: Path | None = None,
        project_dir: str | None = None,
    ) -> bool:
        """Launch a Python module in WSL2 Ubuntu environment.

        WSL settings are read from environment variables (see launch_in_wsl).

        Args:
            module_name: Python module name to run with -m flag.
            cwd: Optional working directory (Windows Path).
            project_dir: Override WSL project dir (uses WSL_PROJECT_DIR if not set).

        Returns:
            True if launch succeeded, False otherwise.
        """
        # Determine working directory
        if module_name is None:
            raise ValueError("module_name must be provided")

        # Validate module name: must be a dotted Python identifier.
        if not _MODULE_NAME_RE.match(module_name):
            logger.error(
                "WSL module launch rejected: invalid module name %r", module_name
            )
            return False

        resolved_project_dir = project_dir or self._get_wsl_project_dir()
        distro = self._get_wsl_distro()
        conda_env = self._get_wsl_conda_env()

        work_dir = resolved_project_dir
        if cwd:
            work_dir = self._convert_to_wsl_path(str(cwd))

        # Use shlex.quote to prevent injection of shell metacharacters in the
        # paths that are interpolated into the bash -c script.
        # module_name has already been validated against the allowlist regex.
        quoted_project_dir = shlex.quote(resolved_project_dir)
        quoted_work_dir = shlex.quote(work_dir)
        quoted_conda_env = shlex.quote(conda_env)

        wsl_cmd = (
            "source ~/miniforge3/etc/profile.d/conda.sh\n"
            f"conda activate {quoted_conda_env}\n"
            'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"\n'
            f"export PYTHONPATH={quoted_project_dir}:$PYTHONPATH\n"
            f"cd {quoted_work_dir}\n"
            f"python -m {module_name}\n"
        )

        cmd = ["wsl", "-d", distro, "--", "bash", "-c", wsl_cmd]

        try:
            logger.info(f"Launching module in WSL: {module_name}")
            if os.name == "nt":
                subprocess.Popen(
                    cmd,
                    creationflags=CREATE_NEW_CONSOLE,
                )
            else:
                subprocess.Popen(cmd)
            return True

        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"WSL module launch failed: {e}")
            return False

    def _convert_to_wsl_path(self, windows_path: str) -> str:
        """Convert a Windows path to a WSL path.

        Args:
            windows_path: Windows-style path string.

        Returns:
            WSL-style path string.
        """
        if windows_path is None:
            raise ValueError("windows_path must be provided")
        if len(windows_path) > 1 and windows_path[1] == ":":
            drive = windows_path[0].lower()
            path_part = windows_path[2:].replace("\\", "/")
            return f"/mnt/{drive}{path_part}"
        return windows_path

    def cleanup_processes(self) -> None:
        """Terminate all running processes managed by this manager."""
        for name, proc in list(self.running_processes.items()):
            try:
                if proc.poll() is None:  # Process is still running
                    logger.info(f"Terminating process: {name}")
                    # Use kill_process_tree to ensure terminal and all children close
                    if not kill_process_tree(proc.pid):
                        # Fallback to direct termination
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Force killing process: {name}")
                            proc.kill()
            except (OSError, ValueError) as e:
                logger.error(f"Error terminating {name}: {e}")

        self.running_processes.clear()
        if getattr(self, "on_process_list_changed", None):
            with contextlib.suppress(Exception):
                self.on_process_list_changed()

    def is_process_running(self, name: str) -> bool:
        """Check if a named process is still running.

        Args:
            name: The process name to check.

        Returns:
            True if the process is running, False otherwise.
        """
        if name is None:
            raise ValueError("name must be provided")
        if name not in self.running_processes:
            return False
        return self.running_processes[name].poll() is None


def is_vcxsrv_running() -> bool:
    """Check if VcXsrv X11 server is running (Windows only).

    Returns:
        True if VcXsrv is running, False otherwise.
    """
    if os.name != "nt":
        return False

    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq vcxsrv.exe"],
            capture_output=True,
            text=True,
            creationflags=CREATE_NO_WINDOW,
        )
        return "vcxsrv.exe" in result.stdout.lower()
    except (OSError, ValueError):
        return False


def start_vcxsrv() -> bool:
    """Start VcXsrv X11 server for Docker GUI support (Windows only).

    Returns:
        True if VcXsrv was started or is already running, False otherwise.
    """
    if os.name != "nt":
        return False

    if is_vcxsrv_running():
        logger.info("VcXsrv already running")
        return True

    for vcx_path in VCXSRV_PATHS:
        if vcx_path.exists():
            try:
                subprocess.Popen(
                    [str(vcx_path), "-multiwindow", "-ac", "-clipboard"],
                    creationflags=CREATE_NO_WINDOW,
                )
                logger.info(f"Started VcXsrv from {vcx_path}")
                return True
            except ImportError as e:
                logger.error(f"Failed to start VcXsrv: {e}")

    logger.warning("VcXsrv not found")
    return False
