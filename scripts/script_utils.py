"""Common utilities and standard lifecycle management for maintenance scripts.

This module provides a robust foundation for repository-level tasks,
standardizing logging, error handling, and common CLI flows.

Principles:
- Orthogonality: Logic is decoupled from entry points.
- DRY: Centralizes recurring patterns found in dozens of scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

# Add project root to path first (script is in scripts/ directory)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.shared.python.logging_config import get_logger, setup_logging

T = TypeVar("T")


def setup_script_logging(name: str) -> logging.Logger:
    """Standardized logging setup for scripts."""
    setup_logging(use_simple_format=True)
    return get_logger(name)


def get_repo_root() -> Path:
    """Get the absolute path to the repository root."""
    return _REPO_ROOT


def run_main(main_func: Callable[[], int | None], logger: logging.Logger) -> None:
    """Standard execution wrapper for script 'main' functions."""
    try:
        exit_code = main_func()
        sys.exit(exit_code if exit_code is not None else 0)
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


def find_python_files(root: Path | str = ".") -> list[Path]:
    """Find source Python files, excluding build and venv artifacts."""
    root_path = Path(root)
    excluded = {".git", "__pycache__", ".venv", "venv", "node_modules", ".tox", "build", "dist"}
    return [
        f for f in root_path.glob("**/*.py")
        if not any(p in f.parts for p in excluded)
    ]


def count_test_files(root: Path | str = ".") -> int:
    """Standardized count of test files using common naming patterns."""
    root_path = Path(root)
    patterns = ["**/test_*.py", "**/*_test.py", "**/tests/*.py"]
    test_files = set()
    for p in patterns:
        test_files.update(root_path.glob(p))
    return len(test_files)


def run_tool_check(cmd: list[str]) -> dict[str, Any]:
    """Execute a tool command and capture status/output."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {"exit_code": res.returncode, "stdout": res.stdout, "stderr": res.stderr}
    except FileNotFoundError:
        return {"exit_code": -1, "stdout": "", "stderr": f"{cmd[0]} not found"}


def check_docs_status(root: Path | str = ".") -> dict[str, bool]:
    """Verify presence of core documentation files."""
    r_path = Path(root)
    return {
        "readme": (r_path / "README.md").exists(),
        "docs_dir": (r_path / "docs").exists(),
        "changelog": (r_path / "CHANGELOG.md").exists(),
    }


def run_pytest(
    path: Path | str = "tests",
    verbose: bool = True,
    markers: str | None = None,
    extra_args: Sequence[str] | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """Run pytest with consistent configuration."""
    if logger:
        logger.info(f"Running tests in {path}...")

    cmd = [sys.executable, "-m", "pytest", str(path)]
    if verbose:
        cmd.append("-v")
    if markers:
        cmd.extend(["-m", markers])
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        if logger:
            logger.error(f"Test execution failed: {e}")
        return False
