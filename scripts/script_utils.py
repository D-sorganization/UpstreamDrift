"""Shared utilities for Golf Modeling Suite scripts.

This module consolidates common patterns used across scripts to address
DRY violations identified by Pragmatic Programmer reviews.

Common Patterns Consolidated:
- Logging setup (delegated to src.shared.python.logging_config)
- Repository root detection (delegated to src.shared.python.path_utils)
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

# Add project root to path first (script is in scripts/ directory)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Import centralized utilities
from src.shared.python.logging_config import (  # noqa: E402
    LogLevel,
    get_logger,
    setup_logging,
)
from src.shared.python.path_utils import get_repo_root  # noqa: E402


def setup_script_logging(
    name: str,
    level: int = logging.INFO,
    format_string: str | None = None,
) -> logging.Logger:
    """Configure logging for a script with consistent formatting.

    This function delegates to the centralized logging_config module.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.
        format_string: Log message format (optional, uses default if None).

    Returns:
        Configured logger instance.
    """
    # Convert int level to LogLevel if needed
    log_level = LogLevel(level) if level in [e.value for e in LogLevel] else level
    setup_logging(level=log_level, format_string=format_string, stream=sys.stdout)
    return get_logger(name)


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
