"""Web application launch helpers for shared GUI launchers."""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

# Exit code conventionally reported for SIGINT/Ctrl-C interrupted runs.
_SIGINT_EXIT_CODE = 130
# Grace period to let the dev server shut down after terminate() before kill().
_SHUTDOWN_GRACE_SECONDS = 10.0
# Bounded readiness-probe parameters for the dev server port.
_READINESS_TIMEOUT_SECONDS = 30.0
_READINESS_POLL_INTERVAL_SECONDS = 0.1


def _resolve_command(command: str) -> str | None:
    """Return the executable path for a command if it is available."""
    return shutil.which(command)


def _verify_command(command: str) -> bool:
    """Return whether a command can run its version probe."""
    executable = _resolve_command(command)
    if executable is None:
        logger.error("Error: %s is not installed or not in PATH", command)
        if command == "node":
            logger.info("Install Node.js from https://nodejs.org/")
        return False

    try:
        subprocess.run(
            [executable, "--version"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Error: %s is not installed or not in PATH", command)
        if command == "node":
            logger.info("Install Node.js from https://nodejs.org/")
        return False

    return True


def _npm_executable() -> str | None:
    """Return the npm executable path after validating Node and npm."""
    for command in ("node", "npm"):
        if not _verify_command(command):
            return None
    return _resolve_command("npm")


def _build_dev_command(
    npm_path: str,
    npm_args: Sequence[str] | None,
) -> list[str]:
    """Build the npm dev-server command."""
    dev_cmd = [npm_path, "run", "dev"]
    if npm_args:
        dev_cmd.extend(npm_args)
    return dev_cmd


def _wait_for_port(
    port: int,
    *,
    host: str = "localhost",
    timeout: float = _READINESS_TIMEOUT_SECONDS,
    poll_interval: float = _READINESS_POLL_INTERVAL_SECONDS,
) -> bool:
    """Block until ``host:port`` accepts a TCP connection or ``timeout`` elapses.

    Returns ``True`` if the port became connectable within the deadline,
    ``False`` otherwise. This replaces a fixed ``time.sleep`` guess so the
    browser is opened only once the dev server is actually listening.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=poll_interval):
                return True
        except OSError:
            time.sleep(poll_interval)
    return False


def _open_browser_later(port: int) -> None:
    """Open the local dev-server URL once the server is accepting connections."""
    if _wait_for_port(port):
        webbrowser.open(f"http://localhost:{port}")
    else:
        logger.warning(
            "Dev server on port %s did not become ready within %.0fs; "
            "not opening browser automatically",
            port,
            _READINESS_TIMEOUT_SECONDS,
        )


def launch_web_app(
    tool_name: str,
    web_dir: Path,
    port: int = 5173,
    auto_open_browser: bool = True,
    npm_args: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
    process_started: Callable[[subprocess.Popen], None] | None = None,
) -> int:
    """Launch a React/Vite web application dev server."""
    if tool_name is None:
        raise ValueError("tool_name must be provided")

    npm_path = _npm_executable()
    if npm_path is None:
        return 1

    if not web_dir.exists():
        logger.error("Error: Web directory not found at %s", web_dir)
        return 1

    if not (web_dir / "node_modules").exists():
        logger.info("Installing dependencies...")
        install_result = subprocess.run(
            [npm_path, "install"],
            cwd=str(web_dir),
            shell=False,
        )
        if install_result.returncode != 0:
            logger.error("Error: Failed to install npm dependencies")
            return 1

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    env["PORT"] = str(port)

    logger.info("Starting %s web application on http://localhost:%s", tool_name, port)

    process = subprocess.Popen(
        _build_dev_command(npm_path, npm_args),
        cwd=str(web_dir),
        env=env,
        shell=False,
    )
    if process_started is not None:
        process_started(process)

    if auto_open_browser:
        threading.Thread(target=_open_browser_later, args=(port,), daemon=True).start()

    try:
        return process.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return _reap_child(process)


def _reap_child(process: subprocess.Popen) -> int:
    """Terminate ``process`` and wait for it, escalating to kill on timeout.

    Postcondition: the child does not outlive this call — it is terminated,
    awaited, and force-killed if it does not exit within the grace period.
    Returns a non-zero exit code (130 for the interrupted run) so the parent
    process manager does not record a clean exit for an aborted shutdown.
    """
    process.terminate()
    try:
        process.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        logger.warning(
            "Dev server did not exit within %.0fs; killing",
            _SHUTDOWN_GRACE_SECONDS,
        )
        process.kill()
        try:
            process.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            logger.error("Dev server could not be killed")
    return _SIGINT_EXIT_CODE


def launch_web_from_gui_info(gui_info: dict[str, object], caller_file: str) -> int:
    """Launch a React web app from a ``GUI_INFO`` dict."""
    if gui_info is None:
        raise ValueError("gui_info must be provided")
    web_cfg = gui_info.get("web", {})
    if not isinstance(web_cfg, dict):
        web_cfg = {}
    tool_name = str(gui_info.get("name", gui_info.get("tool_name", "Unknown")))

    web_path = str(web_cfg.get("path", "web"))
    web_dir = Path(caller_file).parent / web_path
    port = web_cfg.get("port", 5173)
    auto_open = web_cfg.get("auto_open_browser", True)

    return launch_web_app(
        tool_name=tool_name,
        web_dir=web_dir,
        port=int(port),
        auto_open_browser=bool(auto_open),
    )
