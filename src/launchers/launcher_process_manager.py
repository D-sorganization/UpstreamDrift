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

import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from src.shared.python.logging_config import get_logger
from src.shared.python.subprocess_utils import kill_process_tree

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

# VcXsrv paths for Windows X11 support
VCXSRV_PATHS = [
    Path("C:/Program Files/VcXsrv/vcxsrv.exe"),
    Path("C:/Program Files (x86)/VcXsrv/vcxsrv.exe"),
]


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
        self.repo_root = repo_root
        self.running_processes: dict[str, Popen[bytes]] = {}
        self.output_callback = output_callback
        self.use_separate_terminals = use_separate_terminals
        self._output_threads: dict[str, threading.Thread] = {}

    def get_subprocess_env(self) -> dict[str, str]:
        """Get environment variables for subprocess execution.

        Returns:
            Dictionary of environment variables with proper PYTHONPATH.
        """
        env = os.environ.copy()
        repo_root_str = str(self.repo_root)
        src_dir = str(self.repo_root / "src")

        existing_path = env.get("PYTHONPATH", "")
        paths_to_add = []

        if repo_root_str not in existing_path:
            paths_to_add.append(repo_root_str)
        if src_dir not in existing_path:
            paths_to_add.append(src_dir)

        separator = ";" if os.name == "nt" else ":"
        if paths_to_add:
            new_paths = separator.join(paths_to_add)
            env["PYTHONPATH"] = (
                f"{new_paths}{separator}{existing_path}" if existing_path else new_paths
            )

        return env

    def _emit_output(self, name: str, line: str) -> None:
        """Route a line of process output to the callback or logger."""
        if self.output_callback is not None:
            self.output_callback(name, line)
        else:
            logger.info("[%s] %s", name, line)

    def _stream_output(self, name: str, process: subprocess.Popen[bytes]) -> None:
        """Read stdout/stderr from *process* and emit lines until EOF.

        Runs in a daemon thread so the main GUI thread is never blocked.
        """
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
        except Exception as e:
            logger.debug("Output stream ended for %s: %s", name, e)

        return_code = process.wait()
        self._emit_output(name, f"[exited with code {return_code}]")

    def launch_script(
        self,
        name: str,
        script_path: Path,
        cwd: Path,
        env: dict[str, str] | None = None,
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
            process_env = env or self.get_subprocess_env()

            if self.use_separate_terminals:
                # Legacy: each engine gets its own console window
                if os.name == "nt":
                    if keep_terminal_open:
                        cmd_str = f'cmd /k ""{sys.executable}" "{script_path}" & pause"'
                    else:
                        cmd_str = f'cmd /c ""{sys.executable}" "{script_path}""'
                    process = subprocess.Popen(
                        cmd_str,
                        cwd=str(cwd),
                        env=process_env,
                        creationflags=CREATE_NEW_CONSOLE,
                    )
                else:
                    process = subprocess.Popen(
                        [sys.executable, str(script_path)],
                        cwd=str(cwd),
                        env=process_env,
                    )
            else:
                # Unified console: capture output via pipes
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    cwd=str(cwd),
                    env=process_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                # Stream output in a background thread
                t = threading.Thread(
                    target=self._stream_output,
                    args=(name, process),
                    daemon=True,
                )
                t.start()
                self._output_threads[name] = t

            self.running_processes[name] = process
            logger.info(f"Launched {name} (PID: {process.pid})")
            return process

        except Exception as e:
            logger.error(f"Failed to launch {name}: {e}")
            return None

    def launch_module(
        self,
        name: str,
        module_name: str,
        cwd: Path,
        env: dict[str, str] | None = None,
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
            process_env = env or self.get_subprocess_env()

            # Ensure PYTHONPATH is set correctly
            if os.name == "nt":
                current_pythonpath = process_env.get("PYTHONPATH", "")
                repo_root_str = str(self.repo_root)
                src_dir_str = str(self.repo_root / "src")

                paths_to_add = []
                if repo_root_str not in current_pythonpath:
                    paths_to_add.append(repo_root_str)
                if src_dir_str not in current_pythonpath:
                    paths_to_add.append(src_dir_str)

                if paths_to_add:
                    process_env["PYTHONPATH"] = (
                        f"{';'.join(paths_to_add)};{current_pythonpath}"
                    )

            if self.use_separate_terminals:
                # Legacy: each engine gets its own console window
                if os.name == "nt":
                    if keep_terminal_open:
                        cmd_str = (
                            f'cmd /k ""{sys.executable}" -m {module_name} & pause"'
                        )
                    else:
                        cmd_str = f'cmd /c ""{sys.executable}" -m {module_name}"'
                    process = subprocess.Popen(
                        cmd_str,
                        cwd=str(cwd),
                        env=process_env,
                        creationflags=CREATE_NEW_CONSOLE,
                    )
                else:
                    process = subprocess.Popen(
                        [sys.executable, "-m", module_name],
                        cwd=str(cwd),
                        env=process_env,
                    )
            else:
                # Unified console: capture output via pipes
                process = subprocess.Popen(
                    [sys.executable, "-m", module_name],
                    cwd=str(cwd),
                    env=process_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                t = threading.Thread(
                    target=self._stream_output,
                    args=(name, process),
                    daemon=True,
                )
                t.start()
                self._output_threads[name] = t

            self.running_processes[name] = process
            logger.info(f"Launched module {name} (PID: {process.pid})")
            return process

        except Exception as e:
            logger.error(f"Failed to launch {name}: {e}")
            return None

    def launch_in_wsl(
        self,
        script_path: str,
        project_dir: str = "/mnt/c/Users/diete/Repositories/UpstreamDrift",
    ) -> bool:
        """Launch a script in WSL2 Ubuntu environment.

        Args:
            script_path: Windows path to the script.
            project_dir: WSL path to the project directory.

        Returns:
            True if launch succeeded, False otherwise.
        """
        # Convert Windows path to WSL path
        wsl_script_path = self._convert_to_wsl_path(script_path)

        wsl_cmd = f"""
source ~/miniforge3/etc/profile.d/conda.sh
conda activate golf_suite
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="{project_dir}:$PYTHONPATH"
cd "{project_dir}"
python "{wsl_script_path}"
"""

        cmd = ["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-c", wsl_cmd]

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

        except Exception as e:
            logger.error(f"WSL launch failed: {e}")
            return False

    def launch_module_in_wsl(
        self,
        module_name: str,
        cwd: Path | None = None,
        project_dir: str = "/mnt/c/Users/diete/Repositories/UpstreamDrift",
    ) -> bool:
        """Launch a Python module in WSL2 Ubuntu environment.

        Args:
            module_name: Python module name to run with -m flag.
            cwd: Optional working directory (Windows Path).
            project_dir: WSL path to the project directory.

        Returns:
            True if launch succeeded, False otherwise.
        """
        # Determine working directory
        work_dir = project_dir
        if cwd:
            work_dir = self._convert_to_wsl_path(str(cwd))

        wsl_cmd = f"""
source ~/miniforge3/etc/profile.d/conda.sh
conda activate golf_suite
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="{project_dir}:$PYTHONPATH"
cd "{work_dir}"
python -m {module_name}
"""

        cmd = ["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-c", wsl_cmd]

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

        except Exception as e:
            logger.error(f"WSL module launch failed: {e}")
            return False

    def _convert_to_wsl_path(self, windows_path: str) -> str:
        """Convert a Windows path to a WSL path.

        Args:
            windows_path: Windows-style path string.

        Returns:
            WSL-style path string.
        """
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
            except Exception as e:
                logger.error(f"Error terminating {name}: {e}")

        self.running_processes.clear()

    def is_process_running(self, name: str) -> bool:
        """Check if a named process is still running.

        Args:
            name: The process name to check.

        Returns:
            True if the process is running, False otherwise.
        """
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
    except Exception:
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
            except Exception as e:
                logger.error(f"Failed to start VcXsrv: {e}")

    logger.warning("VcXsrv not found")
    return False
