"""Launcher utilities for consolidating common startup script logic.

This module provides shared capabilities for dependency checking, repository
syncing, and environment validation to address DRY violations in launcher scripts.

Principles:
- DRY: Extract common startup logic used by multiple entry points.
- Orthogonality: Decouple environment management from component business logic.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path

from src.shared.python.logging_config import get_logger
from src.shared.python.subprocess_utils import run_command

logger = get_logger(__name__)


def invoke_main(main_func: Callable[[], int | None]) -> None:
    """Standardized entry point for launcher scripts.

    Orthogonality: Decouples the command-line entry/exit mechanics from
    the launcher's operational logic.

    Args:
        main_func: The main function to execute.
    """
    try:
        result = main_func()
        sys.exit(result if result is not None else 0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal launcher error: {e}", exc_info=True)
        sys.exit(1)


def check_python_dependencies(
    required_modules: list[str], install_missing: bool = False
) -> bool:
    """Check if required python modules are installed.

    Args:
        required_modules: List of module names to check.
        install_missing: if True, attempt to pip install missing modules.

    Returns:
        True if all modules are available (or installed), False otherwise.
    """
    missing = []
    for module in required_modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)

    if not missing:
        return True

    logger.warning(f"Missing dependencies: {', '.join(missing)}")

    if install_missing:
        logger.info("Attempting to install missing dependencies...")
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + missing
            result = run_command(cmd)
            if result and result.returncode == 0:
                logger.info("Successfully installed missing dependencies.")
                return True
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")

    logger.error(f"Dependency check failed. Missing: {missing}")
    return False


def git_sync_repository(repo_path: Path | None = None) -> bool:
    """Sync the repository with remote.

    Args:
        repo_path: Path to repository root. Defaults to script parent.

    Returns:
        True if sync succeeded, False otherwise.
    """
    if repo_path is None:
        repo_path = Path(__file__).parent.parent.parent.parent

    logger.info("Syncing repository with remote...")
    try:
        # 1. Fetch
        run_command(["git", "fetch", "--all"], cwd=repo_path)

        # 2. Pull (fast-forward if possible)
        # We don't use --ff-only to allow for quiet merges if safe
        pull_result = run_command(["git", "pull"], cwd=repo_path)

        if pull_result and pull_result.returncode == 0:
            logger.info("Repository sync complete.")
            return True
        else:
            logger.warning("Git pull failed.")
            return False
    except Exception as e:
        logger.warning(f"Git sync failed (might be offline or have conflicts): {e}")
        return False


def get_repo_root() -> Path:
    """Safely resolve the repository root path.

    Returns:
        Path to the repository root directory.
    """
    return Path(__file__).parent.parent.parent.parent.absolute()


def ensure_environment_var(
    name: str, default: str, description: str | None = None
) -> str:
    """Ensure an environment variable is set, using a default if missing.

    Args:
        name: Environment variable name.
        default: Default value if not set.
        description: Optional description for logging.

    Returns:
        The current or default value of the environment variable.
    """
    value = os.getenv(name)
    if value:
        return value

    os.environ[name] = default
    if description:
        logger.info(f"Setting default {name} ({description}): {default}")
    else:
        logger.info(f"Setting default {name}: {default}")

    return default
