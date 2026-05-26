"""Secure subprocess utilities for Golf Modeling Suite.

This module provides secure wrappers around subprocess calls to prevent
command injection and other security vulnerabilities.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Allowed script directories (whitelist approach)
ALLOWED_SCRIPT_DIRECTORIES = [
    "src",
    "engines",
    "launchers",
    "tools",
    "shared",
    "examples",
]

# Allowed executables (whitelist approach).
#
# `wsl`/`wsl.exe` are allowed because `get_docker_cmd()` in
# `src/launchers/docker_manager.py` falls back to `["wsl", "docker"]` on
# Windows hosts where Docker Desktop is not installed but Docker is reachable
# inside WSL. Blocking `wsl` caused the launcher's Docker probe to silently
# report "unavailable" on every WSL-only Windows dev box.
ALLOWED_EXECUTABLES = [
    "python",
    "python3",
    "python.exe",
    "python3.exe",
    "matlab",
    "matlab.exe",
    "docker",
    "docker.exe",
    "wsl",
    "wsl.exe",
    "echo",
]


class SecureSubprocessError(Exception):
    """Exception raised for subprocess security violations."""


def _apply_hidden_window_default(kwargs: dict[str, Any]) -> None:
    """Hide background subprocess console windows on Windows by default."""
    if os.name != "nt" or "creationflags" in kwargs:
        return
    kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def _find_tools_repo_for_security(suite_root: Path) -> Path | None:
    """Find sibling Tools repository path for security validations without importing launchers."""
    env_path = os.environ.get("TOOLS_REPO_PATH")
    if env_path and Path(env_path).is_dir():
        return Path(env_path)
    p = suite_root
    for _ in range(10):
        for candidate in (
            p / "Tools",
            p / "Repositories" / "Tools",
            Path.home() / "Repositories" / "Tools",
        ):
            if candidate.is_dir() and (candidate / "src").is_dir():
                try:
                    # Skip candidate if it is nested inside our repo (e.g. the vendored copy)
                    # to prioritize a true sibling checkout.
                    try:
                        if candidate.is_relative_to(suite_root):
                            continue
                    except (ValueError, AttributeError):
                        if str(suite_root) in str(candidate.resolve()):
                            continue
                except Exception:  # noqa: BLE001
                    pass
                return candidate
        p = p.parent
    return None


def validate_script_path(script_path: Path, suite_root: Path) -> None:
    """Validate that a script path is safe to execute.

    Args:
        script_path: Path to the script to validate
        suite_root: Root directory of the suite

    Raises:
        SecureSubprocessError: If the path is not safe
    """
    try:
        # Resolve to absolute path to prevent path traversal
        abs_script = script_path.resolve()
        abs_suite_root = suite_root.resolve()

        tools_repo = _find_tools_repo_for_security(abs_suite_root)
        in_suite = str(abs_script).startswith(str(abs_suite_root))
        in_tools = tools_repo is not None and str(abs_script).startswith(
            str(tools_repo.resolve())
        )

        # Ensure script is within suite directory or tools directory
        if not in_suite and not in_tools:
            raise SecureSubprocessError(
                f"Script path outside allowed suite/tools directories: {abs_script}"
            )

        # Check if script is in allowed directory
        if in_suite:
            relative_path = abs_script.relative_to(abs_suite_root)
            first_part = relative_path.parts[0] if relative_path.parts else ""
            if first_part not in ALLOWED_SCRIPT_DIRECTORIES:
                raise SecureSubprocessError(
                    f"Script in disallowed directory: {first_part}"
                )
        elif in_tools and tools_repo is not None:
            # Sibling repos like Tools usually have "src" as their root dir for scripts
            relative_path = abs_script.relative_to(tools_repo.resolve())
            first_part = relative_path.parts[0] if relative_path.parts else ""
            # Allow "src" or other standard directories for sibling execution
            if first_part not in ALLOWED_SCRIPT_DIRECTORIES:
                raise SecureSubprocessError(
                    f"Script in disallowed directory of Tools: {first_part}"
                )

        # Ensure file exists and is a file
        if not abs_script.exists():
            raise SecureSubprocessError(f"Script does not exist: {abs_script}")

        if not abs_script.is_file():
            raise SecureSubprocessError(f"Path is not a file: {abs_script}")

        logger.debug(f"Script path validated: {abs_script}")

    except (OSError, ValueError) as e:
        raise SecureSubprocessError(f"Path validation failed: {e}") from e


def validate_executable(executable: str) -> str:
    """Validate that an executable is safe to run.

    Args:
        executable: Name or path of executable

    Returns:
        Validated executable path

    Raises:
        SecureSubprocessError: If executable is not allowed
    """
    # Handle sys.executable specially (always allowed)
    if executable == sys.executable:
        return executable

    # Extract just the executable name
    exec_name = Path(executable).name.lower()

    if exec_name not in ALLOWED_EXECUTABLES:
        logger.error(f"Blocked disallowed executable: {exec_name}")
        raise SecureSubprocessError(f"Executable not allowed: {exec_name}")

    logger.debug(f"Executable validated: {executable}")
    return executable


def secure_popen(  # noqa: C901
    cmd: list[str],
    cwd: Path | str | None = None,
    suite_root: Path | None = None,
    **kwargs: Any,
) -> subprocess.Popen:
    """Securely launch a subprocess with validation.

    Args:
        cmd: Command list (first element is executable)
        cwd: Working directory
        suite_root: Suite root for validation
        **kwargs: Additional arguments for Popen

    Returns:
        Popen process object

    Raises:
        SecureSubprocessError: If command is not safe
    """
    if not cmd:
        raise SecureSubprocessError("Empty command list")

    # Security: Never use shell=True (Checked first)
    if kwargs.get("shell", False):
        raise SecureSubprocessError("shell=True is not allowed for security")

    # Validate executable
    validated_executable = validate_executable(cmd[0])
    validated_cmd = [validated_executable] + cmd[1:]

    # If script path provided, validate it
    if len(cmd) >= 2 and suite_root:
        script_arg = cmd[1]
        # Check if it looks like a script path
        if (
            script_arg.endswith((".py", ".m"))
            or "/" in script_arg
            or "\\" in script_arg
        ):
            try:
                script_path = Path(script_arg)
                if not script_path.is_absolute():
                    # Make relative to cwd if provided, otherwise suite_root
                    base_path = Path(cwd) if cwd else suite_root
                    script_path = base_path / script_path
                validate_script_path(script_path, suite_root)
            except (ValueError, OSError):
                # If path parsing fails, continue (might be a module name)
                pass

    # Validate working directory
    if cwd:
        cwd_path = Path(cwd).resolve()
        if suite_root:
            suite_root_abs = suite_root.resolve()
            tools_repo = _find_tools_repo_for_security(suite_root_abs)
            in_suite = cwd_path.is_relative_to(suite_root_abs)
            in_tools = tools_repo is not None and cwd_path.is_relative_to(
                tools_repo.resolve()
            )
            if not in_suite and not in_tools:
                raise SecureSubprocessError(
                    f"Working directory outside allowed suite/tools directories: {cwd_path}"
                )

    logger.info(f"Launching secure subprocess: {' '.join(validated_cmd)}")
    _apply_hidden_window_default(kwargs)

    try:
        return subprocess.Popen(validated_cmd, cwd=cwd, **kwargs)
    except (OSError, subprocess.SubprocessError) as e:
        logger.error(f"Failed to launch subprocess: {e}")
        raise SecureSubprocessError(f"Subprocess launch failed: {e}") from e


def secure_run(  # noqa: C901
    cmd: list[str],
    cwd: Path | str | None = None,
    suite_root: Path | None = None,
    timeout: float = 30.0,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Securely run a subprocess with validation and timeout.

    Args:
        cmd: Command list
        cwd: Working directory
        suite_root: Suite root for validation
        timeout: Timeout in seconds
        **kwargs: Additional arguments for run

    Returns:
        CompletedProcess result

    Raises:
        SecureSubprocessError: If command is not safe
    """
    if not cmd:
        raise SecureSubprocessError("Empty command list")

    # Security: Never use shell=True (Checked first)
    if kwargs.get("shell", False):
        raise SecureSubprocessError("shell=True is not allowed for security")

    # Validate executable
    validated_executable = validate_executable(cmd[0])
    validated_cmd = [validated_executable] + cmd[1:]

    # If script path provided, validate it
    if len(cmd) >= 2 and suite_root:
        script_arg = cmd[1]
        # Check if it looks like a script path
        if (
            script_arg.endswith((".py", ".m"))
            or "/" in script_arg
            or "\\" in script_arg
        ):
            try:
                script_path = Path(script_arg)
                if not script_path.is_absolute():
                    # Make relative to cwd if provided, otherwise suite_root
                    base_path = Path(cwd) if cwd else suite_root
                    script_path = base_path / script_path
                validate_script_path(script_path, suite_root)
            except (ValueError, OSError):
                # If path parsing fails, continue (might be a module name)
                pass

    # Validate working directory
    if cwd:
        cwd_path = Path(cwd).resolve()
        if suite_root:
            suite_root_abs = suite_root.resolve()
            tools_repo = _find_tools_repo_for_security(suite_root_abs)
            in_suite = cwd_path.is_relative_to(suite_root_abs)
            in_tools = tools_repo is not None and cwd_path.is_relative_to(
                tools_repo.resolve()
            )
            if not in_suite and not in_tools:
                raise SecureSubprocessError(
                    f"Working directory outside allowed suite/tools directories: {cwd_path}"
                )

    logger.info(f"Running secure subprocess: {' '.join(validated_cmd)}")
    _apply_hidden_window_default(kwargs)

    try:
        return subprocess.run(validated_cmd, cwd=cwd, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as e:
        logger.error(f"Subprocess timed out after {timeout}s: {e}")
        raise SecureSubprocessError(f"Subprocess timeout: {e}") from e
    except (OSError, subprocess.SubprocessError) as e:
        logger.error(f"Failed to run subprocess: {e}")
        raise SecureSubprocessError(f"Subprocess run failed: {e}") from e
