"""Secure subprocess utilities for Golf Modeling Suite.

This module provides secure wrappers around subprocess calls to prevent
command injection and other security vulnerabilities.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

# Allowed script directories (whitelist approach)
ALLOWED_SCRIPT_DIRECTORIES = [
    "engines",
    "launchers",
    "tools",
    "shared",
    "examples",
]

# Allowed executables (whitelist approach)
ALLOWED_EXECUTABLES = [
    "python",
    "python3",
    "python.exe",
    "python3.exe",
    "matlab",
    "matlab.exe",
    "docker",
    "docker.exe",
]


class SecureSubprocessError(Exception):
    """Exception raised for subprocess security violations."""


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

        # Ensure script is within suite directory
        if not str(abs_script).startswith(str(abs_suite_root)):
            raise SecureSubprocessError(
                f"Script path outside suite directory: {abs_script}"
            )

        # Check if script is in allowed directory
        relative_path = abs_script.relative_to(abs_suite_root)
        first_part = relative_path.parts[0] if relative_path.parts else ""

        if first_part not in ALLOWED_SCRIPT_DIRECTORIES:
            raise SecureSubprocessError(f"Script in disallowed directory: {first_part}")

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


def secure_popen(
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
            if not cwd_path.is_relative_to(suite_root_abs):
                raise SecureSubprocessError(
                    f"Working directory outside suite: {cwd_path}"
                )

    logger.info(f"Launching secure subprocess: {' '.join(validated_cmd)}")

    try:
        return subprocess.Popen(validated_cmd, cwd=cwd, **kwargs)
    except (OSError, subprocess.SubprocessError) as e:
        logger.error(f"Failed to launch subprocess: {e}")
        raise SecureSubprocessError(f"Subprocess launch failed: {e}") from e


def secure_run(
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
            if not cwd_path.is_relative_to(suite_root_abs):
                raise SecureSubprocessError(
                    f"Working directory outside suite: {cwd_path}"
                )

    logger.info(f"Running secure subprocess: {' '.join(validated_cmd)}")

    try:
        return subprocess.run(validated_cmd, cwd=cwd, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as e:
        logger.error(f"Subprocess timed out after {timeout}s: {e}")
        raise SecureSubprocessError(f"Subprocess timeout: {e}") from e
    except (OSError, subprocess.SubprocessError) as e:
        logger.error(f"Failed to run subprocess: {e}")
        raise SecureSubprocessError(f"Subprocess run failed: {e}") from e
