"""Shared utilities for Golf Modeling Suite scripts.

This module consolidates common patterns used across scripts to address
DRY violations identified by Pragmatic Programmer reviews.

Common Patterns Consolidated:
- Logging setup
- Repository root detection
- PYTHONPATH environment setup
- Subprocess execution with consistent error handling
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory.

    Works from any script location within the repository.

    Returns:
        Path to the repository root.
    """
    # Scripts are in {repo_root}/scripts/
    return Path(__file__).resolve().parent.parent


def setup_script_logging(
    name: str,
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Configure logging for a script with consistent formatting.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.
        format_string: Log message format.

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(name)


def get_pythonpath_env(additional_paths: Sequence[Path] | None = None) -> dict:
    """Get environment dict with PYTHONPATH set correctly.

    Args:
        additional_paths: Additional paths to include in PYTHONPATH.

    Returns:
        Environment dict with PYTHONPATH configured.
    """
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")

    paths = [str(get_repo_root())]
    if additional_paths:
        paths.extend(str(p) for p in additional_paths)

    env["PYTHONPATH"] = os.pathsep.join(paths) + (
        os.pathsep + pythonpath if pythonpath else ""
    )
    return env


def run_command(
    cmd: Sequence[str],
    cwd: Path | None = None,
    env: dict | None = None,
    capture_output: bool = False,
    check: bool = False,
    logger: logging.Logger | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess command with consistent error handling.

    Args:
        cmd: Command and arguments to run.
        cwd: Working directory (defaults to repo root).
        env: Environment variables (defaults to PYTHONPATH-enhanced env).
        capture_output: Whether to capture stdout/stderr.
        check: Whether to raise on non-zero exit.
        logger: Optional logger for command logging.

    Returns:
        CompletedProcess instance.
    """
    if cwd is None:
        cwd = get_repo_root()
    if env is None:
        env = get_pythonpath_env()

    if logger:
        logger.info(f"Executing: {' '.join(str(c) for c in cmd)}")

    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=capture_output,
        check=check,
    )


def run_pytest(
    test_paths: Sequence[str | Path],
    verbose: bool = True,
    markers: str | None = None,
    extra_args: Sequence[str] | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """Run pytest with consistent configuration.

    Args:
        test_paths: Paths to test files or directories.
        verbose: Enable verbose output.
        markers: Pytest marker expression.
        extra_args: Additional pytest arguments.
        logger: Optional logger for status messages.

    Returns:
        True if tests passed, False otherwise.
    """
    cmd = [sys.executable, "-m", "pytest"]
    cmd.extend(str(p) for p in test_paths)

    if verbose:
        cmd.append("-v")
    if markers:
        cmd.extend(["-m", markers])
    if extra_args:
        cmd.extend(extra_args)

    result = run_command(cmd, logger=logger)

    if logger:
        if result.returncode == 0:
            logger.info("Tests PASSED")
        else:
            logger.error(f"Tests FAILED (Exit Code: {result.returncode})")

    return result.returncode == 0
