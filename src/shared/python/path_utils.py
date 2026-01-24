"""Centralized path utilities for the Golf Modeling Suite.

This module provides utilities for constructing paths throughout the
repository, eliminating duplicate Path(__file__).resolve().parents[N]
patterns (DRY principle).

Usage:
    from src.shared.python.path_utils import (
        get_repo_root,
        get_src_root,
        get_simscape_model_path,
        setup_import_paths,
    )

    # Get repository root
    root = get_repo_root()

    # Get source root
    src = get_src_root()

    # Get Simscape model path
    model_path = get_simscape_model_path("3D_Golf_Model")

    # Setup sys.path for imports (typically in test files)
    setup_import_paths()
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_repo_root() -> Path:
    """Get the repository root directory.

    This function caches its result for performance.

    Returns:
        Path to the repository root directory.
    """
    # path_utils.py is at src/shared/python/path_utils.py
    # So repo root is 4 levels up
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def get_src_root() -> Path:
    """Get the src directory.

    Returns:
        Path to the src directory.
    """
    return get_repo_root() / "src"


@lru_cache(maxsize=1)
def get_shared_python_root() -> Path:
    """Get the shared Python utilities directory.

    Returns:
        Path to src/shared/python.
    """
    return get_src_root() / "shared" / "python"


@lru_cache(maxsize=1)
def get_engines_root() -> Path:
    """Get the engines directory.

    Returns:
        Path to src/engines.
    """
    return get_src_root() / "engines"


@lru_cache(maxsize=1)
def get_tests_root() -> Path:
    """Get the tests directory.

    Returns:
        Path to the tests directory.
    """
    return get_repo_root() / "tests"


def get_simscape_model_path(model_name: str) -> Path:
    """Get the path to a Simscape Multibody Model's Python source directory.

    Args:
        model_name: Name of the model ("2D_Golf_Model" or "3D_Golf_Model")

    Returns:
        Path to the model's Python source directory.

    Raises:
        ValueError: If model_name is not a valid Simscape model.

    Example:
        >>> get_simscape_model_path("3D_Golf_Model")
        PosixPath('.../src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src')
    """
    valid_models = {"2D_Golf_Model", "3D_Golf_Model"}
    if model_name not in valid_models:
        raise ValueError(
            f"Invalid model name: {model_name}. Must be one of: {valid_models}"
        ) from None

    return (
        get_engines_root() / "Simscape_Multibody_Models" / model_name / "python" / "src"
    )


def get_physics_engine_path(engine_name: str) -> Path:
    """Get the path to a physics engine directory.

    Args:
        engine_name: Name of the engine (mujoco, drake, pinocchio, etc.)

    Returns:
        Path to the engine directory.

    Example:
        >>> get_physics_engine_path("mujoco")
        PosixPath('.../src/engines/physics_engines/mujoco')
    """
    return get_engines_root() / "physics_engines" / engine_name


def get_physics_engine_python_path(engine_name: str) -> Path:
    """Get the path to a physics engine's Python directory.

    Args:
        engine_name: Name of the engine (mujoco, drake, pinocchio, etc.)

    Returns:
        Path to the engine's Python directory.

    Example:
        >>> get_physics_engine_python_path("mujoco")
        PosixPath('.../src/engines/physics_engines/mujoco/python')
    """
    return get_physics_engine_path(engine_name) / "python"


def setup_import_paths(
    *,
    include_repo_root: bool = True,
    include_src: bool = True,
    include_engines: bool = False,
    additional_paths: list[Path | str] | None = None,
) -> list[Path]:
    """Setup sys.path for imports across the Golf Modeling Suite.

    This function consolidates the common pattern of adding project paths
    to sys.path, eliminating duplicate code in test files and scripts.

    Args:
        include_repo_root: Whether to add the repository root to sys.path.
        include_src: Whether to add the src directory to sys.path.
        include_engines: Whether to add the engines directory to sys.path.
        additional_paths: Additional paths to add to sys.path.

    Returns:
        List of paths that were added to sys.path.

    Example:
        # In a test file:
        from src.shared.python.path_utils import setup_import_paths
        setup_import_paths()

        # With additional paths:
        setup_import_paths(additional_paths=["/custom/path"])

        # For engine-specific tests:
        setup_import_paths(include_engines=True)
    """
    added_paths: list[Path] = []

    if include_repo_root:
        repo_root = get_repo_root()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
            added_paths.append(repo_root)

    if include_src:
        src_root = get_src_root()
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
            added_paths.append(src_root)

    if include_engines:
        engines_root = get_engines_root()
        if str(engines_root) not in sys.path:
            sys.path.insert(0, str(engines_root))
            added_paths.append(engines_root)

    if additional_paths:
        for path in additional_paths:
            path_obj = Path(path) if isinstance(path, str) else path
            if str(path_obj) not in sys.path:
                sys.path.insert(0, str(path_obj))
                added_paths.append(path_obj)

    return added_paths


def get_ancestor_path_from_file(file_path: str | Path, levels_up: int = 0) -> Path:
    """Get an ancestor directory from a file path.

    This is a utility to help refactor existing patterns that use
    ``Path(__file__).resolve().parents[N]``.

    Args:
        file_path: The ``__file__`` of the calling module (or any file path).
        levels_up: Number of directory levels to go up (0 for the immediate parent).

    Returns:
        The resolved ancestor path at ``levels_up``.

    Example:
        # Instead of: PROJECT_ROOT = Path(__file__).resolve().parents[3]
        # Use: PROJECT_ROOT = get_ancestor_path_from_file(__file__, 3)
    """
    return Path(file_path).resolve().parents[levels_up]


def get_project_root_from_file(file_path: str | Path, levels_up: int = 0) -> Path:
    """Deprecated alias for :func:`get_ancestor_path_from_file`.

    .. deprecated::
        The name of this function is misleading because it returns an arbitrary
        ancestor directory based on ``levels_up``, not necessarily a project root.
        New code should use :func:`get_ancestor_path_from_file` instead.

    Args:
        file_path: The ``__file__`` of the calling module.
        levels_up: Number of directory levels to go up.

    Returns:
        The resolved ancestor path.
    """
    return get_ancestor_path_from_file(file_path=file_path, levels_up=levels_up)
