"""Subprocess utilities for eliminating subprocess management duplication.

This module provides reusable subprocess patterns to eliminate repeated
process management code.

Usage:
    from src.shared.python.security.subprocess_utils import (
        ProcessManager,
        run_command,
        run_command_async,
    )

    # Run command synchronously
    result = run_command(["python", "script.py"])

    # Manage multiple processes
    manager = ProcessManager()
    manager.start("server", ["python", "server.py"])
    manager.stop("server")
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from src.shared.python import SUITE_ROOT
from src.shared.python.core.error_decorators import log_errors
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.security.secure_subprocess import (
    secure_run,
    secure_popen,
    SecureSubprocessError,
)

logger = get_logger(__name__)

# Default timeout for subprocess operations (5 minutes)
# This ensures long-running processes don't hang indefinitely while allowing
# reasonable time for complex operations like model loading or simulation runs.
DEFAULT_SUBPROCESS_TIMEOUT: float = 300.0


def _default_suite_root() -> Path:
    """Return the suite root used to validate background launches.

    ``secure_popen``/``secure_run`` skip script-path *and* working-directory
    validation entirely when ``suite_root`` is falsy -- the allowlist on the
    executable name still applies, but directory-traversal protection is inert
    (issue #9221; PR #9216 landed the executable allowlist only). Both
    ``secure_popen`` call sites in this module therefore supply a root.

    ``SUITE_ROOT`` (``src/shared/python/__init__.py``) is the repository's
    existing canonical constant for this and resolves to the checkout root,
    matching the ``.git``-based project-root rule established by #9224.
    """
    return SUITE_ROOT


def _hidden_window_kwargs() -> dict[str, Any]:
    """Return Windows-only flags for invisible background processes."""
    if os.name != "nt":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


@log_errors("Command execution failed", reraise=False, default_return=None)
def run_command(
    cmd: list[str],
    cwd: str | Path | None = None,
    timeout: float | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess | None:
    """Run command synchronously with error handling.

    Args:
        cmd: Command and arguments as list
        cwd: Working directory
        timeout: Timeout in seconds. If None, uses DEFAULT_SUBPROCESS_TIMEOUT (300s).
            Pass a specific value to override, or use a very large value for
            effectively unlimited execution time.
        capture_output: Whether to capture stdout/stderr

    Returns:
        CompletedProcess object or None if failed

    Example:
        result = run_command(["python", "--version"])
        if result and result.returncode == 0:
            logger.info(result.stdout)
    """
    if cmd is None:
        raise ValueError("cmd must be provided")
    effective_timeout = timeout if timeout is not None else DEFAULT_SUBPROCESS_TIMEOUT
    logger.debug(f"Running command: {' '.join(cmd)} (timeout={effective_timeout}s)")

    result = secure_run(
        cmd,
        cwd=str(cwd) if cwd else None,
        timeout=effective_timeout,
        capture_output=capture_output,
    )

    if result.returncode != 0:
        logger.warning(f"Command failed with code {result.returncode}: {' '.join(cmd)}")

    return result


class ProcessManager:
    """Manager for multiple background processes.

    Example:
        manager = ProcessManager()

        # Start processes
        manager.start("server", ["python", "server.py"])
        manager.start("worker", ["python", "worker.py"])

        # Check status
        if manager.is_running("server"):
            logger.info("Server is running")

        # Stop processes
        manager.stop("server")
        manager.stop_all()
    """

    def __init__(self) -> None:
        """Initialize process manager."""
        self.processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def start(
        self,
        name: str,
        cmd: list[str],
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        suite_root: str | Path | None = None,
    ) -> bool:
        """Start a background process.

        Args:
            name: Process identifier
            cmd: Command and arguments
            cwd: Working directory
            env: Environment variables
            suite_root: Root the script path and working directory must live
                under (or under a trusted sibling provider checkout).
                Defaults to :func:`_default_suite_root`. A launch rejected by
                that validation is logged and returns ``False`` -- the same
                contract as any other launch failure.

        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            # Check if already running
            if name in self.processes:
                if self.is_running(name):
                    logger.warning(f"Process '{name}' is already running")
                    return False
                # Clean up dead process
                del self.processes[name]

            try:
                logger.info(f"Starting process '{name}': {' '.join(cmd)}")

                process = secure_popen(
                    cmd,
                    cwd=str(cwd) if cwd else None,
                    suite_root=(
                        Path(suite_root) if suite_root else _default_suite_root()
                    ),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **_hidden_window_kwargs(),
                )

                self.processes[name] = process
                return True

            except (
                FileNotFoundError,
                PermissionError,
                OSError,
                SecureSubprocessError,
            ) as e:
                logger.error(f"Failed to start process '{name}': {e}")
                return False

    def stop(self, name: str, timeout: float = 5.0) -> bool:
        """Stop a background process.

        Args:
            name: Process identifier
            timeout: Timeout for graceful shutdown

        Returns:
            True if stopped successfully, False otherwise
        """
        with self._lock:
            if name not in self.processes:
                logger.warning(f"Process '{name}' not found")
                return False

            process = self.processes[name]

            try:
                logger.info(f"Stopping process '{name}'")

                # Try graceful termination first
                process.terminate()

                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Force kill if termination times out
                    logger.warning(
                        f"Process '{name}' did not terminate gracefully, killing"
                    )
                    process.kill()
                    process.wait()

                del self.processes[name]
                logger.info(f"Process '{name}' stopped")
                return True

            except (OSError, ValueError) as e:
                logger.error(f"Failed to stop process '{name}': {e}")
                return False

    def is_running(self, name: str) -> bool:
        """Check if process is running.

        Args:
            name: Process identifier

        Returns:
            True if running, False otherwise
        """
        if name is None:
            raise ValueError("name must be provided")
        if name not in self.processes:
            return False

        process = self.processes[name]
        return process.poll() is None

    def get_output(self, name: str) -> tuple[str, str]:
        """Get stdout and stderr from process.

        Args:
            name: Process identifier

        Returns:
            Tuple of (stdout, stderr) as strings
        """
        if name is None:
            raise ValueError("name must be provided")
        if name not in self.processes:
            return "", ""

        process = self.processes[name]

        try:
            # Non-blocking read
            stdout = process.stdout.read().decode() if process.stdout else ""
            stderr = process.stderr.read().decode() if process.stderr else ""
            return stdout, stderr
        except (FileNotFoundError, OSError) as e:
            logger.error(f"Failed to read output from '{name}': {e}")
            return "", ""

    def stop_all(self, timeout: float = 5.0) -> None:
        """Stop all managed processes.

        Args:
            timeout: Timeout for each process
        """
        if timeout is None:
            raise ValueError("timeout must be provided")
        logger.info("Stopping all processes")

        # Get list of names to avoid modifying dict during iteration
        names = list(self.processes.keys())

        for name in names:
            self.stop(name, timeout)

    def list_processes(self) -> dict[str, bool]:
        """List all processes and their running status.

        Returns:
            Dictionary mapping process names to running status
        """
        return {name: self.is_running(name) for name in self.processes}

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop_all()


class CommandRunner:
    """Utility for running commands with common patterns.

    Example:
        runner = CommandRunner(cwd="/path/to/project")

        # Run with output capture
        result = runner.run(["python", "script.py"])

        # Run in background
        runner.run_async(["python", "server.py"])

        # Run with retry
        result = runner.run_with_retry(["flaky_command"], max_attempts=3)
    """

    def __init__(
        self,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        suite_root: str | Path | None = None,
    ) -> None:
        """Initialize command runner.

        Args:
            cwd: Default working directory
            env: Default environment variables
            suite_root: Root that :meth:`run_async` launches must stay within
                (or within a trusted sibling provider checkout). Defaults to
                :func:`_default_suite_root`.
        """
        self.cwd = cwd
        self.env = env
        self.suite_root = Path(suite_root) if suite_root else _default_suite_root()

    def run(
        self,
        cmd: list[str],
        timeout: float | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess | None:
        """Run command synchronously.

        Args:
            cmd: Command and arguments
            timeout: Timeout in seconds
            check: Raise exception on non-zero exit code

        Returns:
            CompletedProcess object or None if failed
        """
        result = run_command(cmd, cwd=self.cwd, timeout=timeout)

        if check and result and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )

        return result

    def run_async(self, cmd: list[str]) -> subprocess.Popen | None:
        """Run command asynchronously.

        Args:
            cmd: Command and arguments

        Returns:
            Popen object or None if failed, including when the script path or
            working directory is rejected as outside ``self.suite_root``.
        """
        try:
            logger.debug(f"Running async command: {' '.join(cmd)}")

            process = secure_popen(
                cmd,
                cwd=str(self.cwd) if self.cwd else None,
                suite_root=self.suite_root,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **_hidden_window_kwargs(),
            )

            return process

        except (
            FileNotFoundError,
            PermissionError,
            OSError,
            SecureSubprocessError,
        ) as e:
            logger.error(f"Failed to run async command: {e}")
            return None

    def run_with_retry(
        self,
        cmd: list[str],
        max_attempts: int = 3,
        delay: float = 1.0,
    ) -> subprocess.CompletedProcess | None:
        """Run command with retry on failure.

        Args:
            cmd: Command and arguments
            max_attempts: Maximum number of attempts
            delay: Delay between attempts in seconds

        Returns:
            CompletedProcess object or None if failed after all attempts
        """
        if cmd is None:
            raise ValueError("cmd must be provided")
        for attempt in range(max_attempts):
            result = self.run(cmd)

            if result and result.returncode == 0:
                return result

            if attempt < max_attempts - 1:
                logger.warning(
                    f"Command failed (attempt {attempt + 1}/{max_attempts}), "
                    f"retrying in {delay}s"
                )
                time.sleep(delay)

        logger.error(f"Command failed after {max_attempts} attempts")
        return None


def kill_process_tree(pid: int, timeout: float = 5.0) -> bool:
    """Kill process and all its children.

    Args:
        pid: Process ID
        timeout: Timeout for graceful shutdown

    Returns:
        True if successful, False otherwise
    """
    try:
        import psutil

        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # Terminate children first
        for child in children:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.terminate()

        # Terminate parent
        parent.terminate()

        # Wait for termination
        gone, alive = psutil.wait_procs([parent] + children, timeout=timeout)

        # Kill any remaining processes
        for p in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                p.kill()

        return True

    except ImportError:
        logger.warning("psutil not installed, using basic kill")
        try:
            import os
            import signal

            os.kill(pid, signal.SIGTERM)
            return True
        except OSError as e:
            logger.error(f"Failed to kill process {pid}: {e}")
            return False

    except (RuntimeError, TypeError, AttributeError) as e:
        logger.error(f"Failed to kill process tree {pid}: {e}")
        return False
