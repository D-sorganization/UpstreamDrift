"""Import utilities for checking module availability and versions.

This module provides reusable import patterns for dependency checking.

Usage:
    from src.shared.python.data_io.import_utils import ensure_imports

    available = ensure_imports("numpy", "matplotlib")
"""

from __future__ import annotations

from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


def ensure_imports(*modules: str) -> dict[str, bool]:
    """Check if modules can be imported.

    Args:
        *modules: Module names to check

    Returns:
        Dictionary mapping module names to availability

    Example:
        available = ensure_imports("numpy", "matplotlib", "torch")
        if not available["torch"]:
            print("PyTorch not available")
    """
    results = {}

    for module in modules:
        try:
            __import__(module)
            results[module] = True
            logger.debug(f"Module available: {module}")
        except ImportError:
            results[module] = False
            logger.debug(f"Module not available: {module}")

    return results


def lazy_import(module_name: str) -> Any:
    """Lazily import module.

    Args:
        module_name: Name of module to import

    Returns:
        Imported module

    Raises:
        ImportError: If module cannot be imported

    Example:
        np = lazy_import("numpy")
    """
    try:
        module = __import__(module_name)
        logger.debug(f"Lazily imported: {module_name}")
        return module
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {e}")
        raise


def import_from(module_name: str, *names: str) -> tuple[Any, ...]:
    """Import specific names from module.

    Args:
        module_name: Module to import from
        *names: Names to import

    Returns:
        Tuple of imported objects

    Example:
        Path, os = import_from("pathlib", "Path"), import_from("os")
    """
    module = __import__(module_name, fromlist=names)
    results = tuple(getattr(module, name) for name in names)
    logger.debug(f"Imported {names} from {module_name}")
    return results if len(results) > 1 else results[0]


def check_optional_dependency(
    module_name: str,
    feature_name: str | None = None,
) -> bool:
    """Check if optional dependency is available.

    Args:
        module_name: Name of module to check
        feature_name: Name of feature requiring module (for error message)

    Returns:
        True if available, False otherwise

    Example:
        if check_optional_dependency("torch", "GPU acceleration"):
            use_gpu()
    """
    try:
        __import__(module_name)
        logger.debug(f"Optional dependency available: {module_name}")
        return True
    except ImportError:
        if feature_name:
            logger.info(
                f"Optional dependency {module_name} not available. "
                f"{feature_name} will be disabled."
            )
        else:
            logger.info(f"Optional dependency {module_name} not available.")
        return False


def get_module_version(module_name: str) -> str | None:
    """Get version of installed module.

    Args:
        module_name: Name of module

    Returns:
        Version string or None if not available

    Example:
        version = get_module_version("numpy")
        print(f"NumPy version: {version}")
    """
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", None)
        if version:
            logger.debug(f"{module_name} version: {version}")
        return version
    except ImportError:
        logger.debug(f"Module {module_name} not available")
        return None


def check_minimum_version(
    module_name: str,
    minimum_version: str,
) -> bool:
    """Check if module meets minimum version requirement.

    Args:
        module_name: Name of module
        minimum_version: Minimum required version

    Returns:
        True if version is sufficient, False otherwise

    Example:
        if check_minimum_version("numpy", "1.20.0"):
            use_new_api()
    """
    from packaging import version

    current_version = get_module_version(module_name)
    if current_version is None:
        return False

    try:
        meets_requirement = version.parse(current_version) >= version.parse(
            minimum_version
        )
        if meets_requirement:
            logger.debug(
                f"{module_name} {current_version} meets minimum {minimum_version}"
            )
        else:
            logger.warning(
                f"{module_name} {current_version} does not meet minimum {minimum_version}"
            )
        return meets_requirement
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Failed to compare versions: {e}")
        return False
