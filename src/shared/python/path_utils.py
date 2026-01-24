"""Centralized path utilities for the Golf Modeling Suite.

This module provides utilities for constructing paths throughout the
repository, eliminating duplicate Path(__file__).resolve().parents[N]
patterns (DRY principle).

Usage:
    from src.shared.python.path_utils import (
        get_repo_root,
        get_src_root,
        get_simscape_model_path,
    )

    # Get repository root
    root = get_repo_root()

    # Get source root
    src = get_src_root()

    # Get Simscape model path
    model_path = get_simscape_model_path("3D_Golf_Model")
"""

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
        get_engines_root()
        / "Simscape_Multibody_Models"
        / model_name
        / "python"
        / "src"
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
