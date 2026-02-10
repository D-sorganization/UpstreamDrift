"""Path utilities for eliminating path resolution duplication.

This module provides reusable path resolution patterns.

Usage:
    from src.shared.python.path_utils import get_repo_root, get_src_root

    repo_root = get_repo_root()
    src_root = get_src_root()
"""

from __future__ import annotations

from pathlib import Path

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

# Cache for path resolutions
_REPO_ROOT: Path | None = None
_SRC_ROOT: Path | None = None


def get_repo_root() -> Path:
    """Get repository root directory.

    Returns:
        Path to repository root

    Example:
        repo_root = get_repo_root()
        data_dir = repo_root / "data"
    """
    global _REPO_ROOT
    if _REPO_ROOT is None:
        # Start from this file and go up to find repo root
        current = Path(__file__).resolve()
        while current.parent != current:
            if (current / ".git").exists() or (current / "pyproject.toml").exists():
                _REPO_ROOT = current
                logger.debug(f"Repository root: {_REPO_ROOT}")
                break
            current = current.parent
        else:
            # Fallback: assume standard structure
            _REPO_ROOT = Path(__file__).resolve().parents[3]
            logger.warning(f"Could not find .git, using fallback: {_REPO_ROOT}")

    return _REPO_ROOT


def get_src_root() -> Path:
    """Get src directory root.

    Returns:
        Path to src directory

    Example:
        src_root = get_src_root()
        shared_dir = src_root / "shared"
    """
    global _SRC_ROOT
    if _SRC_ROOT is None:
        _SRC_ROOT = get_repo_root() / "src"
        logger.debug(f"Source root: {_SRC_ROOT}")

    return _SRC_ROOT


def get_tests_root() -> Path:
    """Get tests directory root.

    Returns:
        Path to tests directory

    Example:
        tests_root = get_tests_root()
        fixtures_dir = tests_root / "fixtures"
    """
    return get_repo_root() / "tests"


def get_data_dir() -> Path:
    """Get data directory.

    Returns:
        Path to data directory

    Example:
        data_dir = get_data_dir()
        trajectory_file = data_dir / "trajectory.csv"
    """
    return get_repo_root() / "data"


def get_output_dir() -> Path:
    """Get output directory.

    Returns:
        Path to output directory

    Example:
        output_dir = get_output_dir()
        results_file = output_dir / "results.json"
    """
    output_dir = get_repo_root() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_docs_dir() -> Path:
    """Get documentation directory.

    Returns:
        Path to docs directory

    Example:
        docs_dir = get_docs_dir()
        api_docs = docs_dir / "api"
    """
    return get_repo_root() / "docs"


def get_engines_dir() -> Path:
    """Get engines directory.

    Returns:
        Path to engines directory

    Example:
        engines_dir = get_engines_dir()
        mujoco_dir = engines_dir / "physics_engines" / "mujoco"
    """
    return get_repo_root() / "engines"


def get_shared_dir() -> Path:
    """Get shared directory.

    Returns:
        Path to shared directory

    Example:
        shared_dir = get_shared_dir()
        models_dir = shared_dir / "models"
    """
    return get_repo_root() / "shared"


def ensure_directory(path: Path | str) -> Path:
    """Ensure directory exists, creating if necessary.

    Args:
        path: Path to directory

    Returns:
        Path object

    Example:
        output_dir = ensure_directory("output/results")
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path_obj}")
    return path_obj


def get_relative_path(path: Path | str, base: Path | str | None = None) -> Path:
    """Get relative path from base directory.

    Args:
        path: Path to make relative
        base: Base directory (defaults to repo root)

    Returns:
        Relative path

    Example:
        rel_path = get_relative_path("/path/to/repo/src/file.py")
        # Returns: src/file.py
    """
    path_obj = Path(path).resolve()
    base_obj = Path(base).resolve() if base else get_repo_root()

    try:
        return path_obj.relative_to(base_obj)
    except ValueError:
        # Path is not relative to base
        return path_obj


def find_file_in_parents(
    filename: str,
    start_path: Path | str | None = None,
    max_levels: int = 5,
) -> Path | None:
    """Find file in parent directories.

    Args:
        filename: Name of file to find
        start_path: Starting directory (defaults to current file location)
        max_levels: Maximum number of parent levels to search

    Returns:
        Path to file if found, None otherwise

    Example:
        pyproject = find_file_in_parents("pyproject.toml")
    """
    if start_path is None:
        current = Path(__file__).resolve().parent
    else:
        current = Path(start_path).resolve()

    for _ in range(max_levels):
        candidate = current / filename
        if candidate.exists():
            logger.debug(f"Found {filename} at {candidate}")
            return candidate

        if current.parent == current:
            break
        current = current.parent

    logger.debug(f"Could not find {filename} in parent directories")
    return None


def get_shared_python_root() -> Path:
    """Get the shared python directory root.

    Returns:
        Path to src/shared/python directory

    Example:
        shared_python = get_shared_python_root()
    """
    return get_src_root() / "shared" / "python"


def get_mujoco_python_root() -> Path:
    """Get the MuJoCo python directory root.

    Returns:
        Path to mujoco python directory

    Example:
        mujoco_python = get_mujoco_python_root()
    """
    return get_src_root() / "engines" / "physics_engines" / "mujoco" / "python"


def setup_import_paths(additional_paths: list[str | Path] | None = None) -> None:
    """Set up Python import paths for the Golf Modeling Suite.

    This function adds necessary directories to sys.path to enable
    proper imports across the codebase. It should be called at the
    start of scripts that need access to shared modules.

    This is a DRY replacement for scattered sys.path.insert calls.

    Args:
        additional_paths: Optional list of additional paths to add to sys.path

    Example:
        from src.shared.python.path_utils import setup_import_paths
        setup_import_paths()

        # With additional paths
        setup_import_paths(additional_paths=["/custom/path"])
    """
    import sys

    paths_to_add = [
        str(get_src_root()),
        str(get_shared_python_root()),
        str(get_mujoco_python_root()),
        str(get_mujoco_python_root() / "mujoco_humanoid_golf"),
    ]

    # Add any additional paths specified
    if additional_paths:
        paths_to_add.extend(str(p) for p in additional_paths)

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            logger.debug(f"Added to sys.path: {path}")


def get_simscape_model_path(model_name: str = "3D_Golf_Model") -> Path:
    """Get the path to Simscape model python/src directory.

    Args:
        model_name: Name of the model directory (default: "3D_Golf_Model")

    Returns:
        Path to Simscape model python/src directory for imports

    Example:
        simscape_path = get_simscape_model_path()
        custom_path = get_simscape_model_path("Custom_Model")
    """
    return (
        get_src_root()
        / "engines"
        / "Simscape_Multibody_Models"
        / model_name
        / "python"
        / "src"
    )


def get_pinocchio_python_root() -> Path:
    """Get the Pinocchio python directory root.

    Returns:
        Path to pinocchio python directory

    Example:
        pinocchio_python = get_pinocchio_python_root()
    """
    return get_src_root() / "engines" / "physics_engines" / "pinocchio" / "python"


def get_drake_python_root() -> Path:
    """Get the Drake python directory root.

    Returns:
        Path to drake python directory

    Example:
        drake_python = get_drake_python_root()
    """
    return get_src_root() / "engines" / "physics_engines" / "drake" / "python"
