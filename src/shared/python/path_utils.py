"""Centralized path utilities for the Golf Modeling Suite.

This module provides standardized path construction to eliminate duplicate
Path(__file__).resolve().parents[N] patterns across the codebase.

Refactored to follow DRY principle - single source of truth for paths.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

# Cache the repository root to avoid repeated computation
_REPO_ROOT: Path | None = None


@lru_cache(maxsize=1)
def get_repo_root() -> Path:
    """Get the repository root directory.

    Returns:
        Path to the repository root (Golf_Modeling_Suite directory).

    Raises:
        RuntimeError: If repository root cannot be determined.
    """
    global _REPO_ROOT

    if _REPO_ROOT is not None:
        return _REPO_ROOT

    # Start from this file's location and walk up to find repo root
    current = Path(__file__).resolve()

    # Walk up until we find a directory containing pyproject.toml or .git
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            _REPO_ROOT = parent
            return parent

    # Fallback: assume standard structure (this file is in src/shared/python/)
    fallback = Path(__file__).resolve().parents[3]
    if fallback.exists():
        _REPO_ROOT = fallback
        return fallback

    raise RuntimeError(
        "Could not determine repository root. "
        "Ensure pyproject.toml or .git exists in the repo root."
    )


def get_src_root() -> Path:
    """Get the src directory path.

    Returns:
        Path to the src directory.
    """
    return get_repo_root() / "src"


def get_shared_python_root() -> Path:
    """Get the shared Python modules directory.

    Returns:
        Path to src/shared/python directory.
    """
    return get_src_root() / "shared" / "python"


def get_tests_root() -> Path:
    """Get the tests directory path.

    Returns:
        Path to the tests directory.
    """
    return get_repo_root() / "tests"


def get_scripts_root() -> Path:
    """Get the scripts directory path.

    Returns:
        Path to the scripts directory.
    """
    return get_repo_root() / "scripts"


def get_data_root() -> Path:
    """Get the data directory path.

    Returns:
        Path to the data directory.
    """
    return get_repo_root() / "data"


def get_models_root() -> Path:
    """Get the models directory path.

    Returns:
        Path to the models directory.
    """
    return get_repo_root() / "models"


def get_docs_root() -> Path:
    """Get the docs directory path.

    Returns:
        Path to the docs directory.
    """
    return get_repo_root() / "docs"


def ensure_repo_in_path() -> None:
    """Ensure the repository root is in sys.path for imports.

    This centralizes the sys.path manipulation that was scattered
    across many test and script files.
    """
    repo_root = get_repo_root()
    repo_str = str(repo_root)

    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def ensure_src_in_path() -> None:
    """Ensure the src directory is in sys.path for imports."""
    src_root = get_src_root()
    src_str = str(src_root)

    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def get_engine_path(engine_name: str) -> Path:
    """Get the path to a specific physics engine directory.

    Args:
        engine_name: Name of the engine (mujoco, pinocchio, drake, opensim, myosuite).

    Returns:
        Path to the engine's python directory.

    Raises:
        ValueError: If engine_name is not recognized.
    """
    valid_engines = {"mujoco", "pinocchio", "drake", "opensim", "myosuite"}
    if engine_name.lower() not in valid_engines:
        raise ValueError(
            f"Unknown engine: {engine_name}. Valid engines: {valid_engines}"
        ) from None

    return (
        get_src_root() / "engines" / "physics_engines" / engine_name.lower() / "python"
    )


def relative_to_repo(path: Path | str) -> Path:
    """Convert an absolute path to be relative to repo root.

    Args:
        path: Path to convert.

    Returns:
        Path relative to repository root.
    """
    path = Path(path).resolve()
    try:
        return path.relative_to(get_repo_root())
    except ValueError:
        # Path is not relative to repo root
        return path


# Aliases for backward compatibility
REPO_ROOT = property(lambda self: get_repo_root())
PROJECT_ROOT = property(lambda self: get_repo_root())
ROOT_DIR = property(lambda self: get_repo_root())
