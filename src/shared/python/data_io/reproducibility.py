"""Reproducibility utilities for deterministic scientific computations.

This module centralizes random seed management across the codebase,
addressing DRY violations identified in Pragmatic Programmer reviews.

Usage:
    from src.shared.python.reproducibility import set_seeds, DEFAULT_SEED

    # Set all random seeds
    set_seeds(42)

    # Use with context manager for timed operations
    with log_execution_time("matrix_computation"):
        result = heavy_computation()
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.engine_availability import PYTORCH_AVAILABLE
from src.shared.python.logging_config import get_logger

if TYPE_CHECKING:
    pass

if PYTORCH_AVAILABLE:
    import torch

logger = get_logger(__name__)

# Reproducibility constants
DEFAULT_SEED: int = 42  # Answer to everything
MAX_SEED: int = np.iinfo(np.uint32).max


def set_seeds(seed: int = DEFAULT_SEED, *, validate: bool = True) -> None:
    """Set random seeds for reproducible computations.

    Sets seeds for Python's random module, NumPy's random generator,
    and PyTorch if available. This ensures deterministic behavior
    across the entire computation pipeline.

    Args:
        seed: Random seed value (default: 42)
        validate: If True, validate seed is within valid range

    Raises:
        ValueError: If validate=True and seed is out of valid range

    Example:
        from src.shared.python.reproducibility import set_seeds

        # At the start of your script/experiment
        set_seeds(42)

        # Now all random operations are deterministic
    """
    if validate and not (0 <= seed <= MAX_SEED):
        raise ValueError(f"Seed must be between 0 and {MAX_SEED}, got {seed}")

    # Python's random module
    random.seed(seed)

    # NumPy - use both legacy and new API for compatibility
    np.random.seed(seed)  # Legacy API (still widely used)
    np.random.default_rng(seed)  # New API

    # PyTorch seeds if available
    if PYTORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.debug("PyTorch seeds set: %d", seed)

    logger.info("All random seeds set to: %d", seed)


@contextmanager
def log_execution_time(
    operation_name: str,
    logger_obj: logging.Logger | None = None,
) -> Generator[None, None, None]:
    """Context manager to log the duration of an operation.

    Useful for profiling and telemetry in scientific computations.

    Args:
        operation_name: Logical name of the operation being timed
        logger_obj: Specific logger to use, or module logger if None

    Yields:
        None

    Example:
        with log_execution_time("forward_kinematics"):
            result = compute_fk(robot, q)
    """
    logr = logger_obj or logger
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logr.info(
            "Telemetry: %s took %.4f seconds",
            operation_name,
            duration,
        )


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Get a NumPy random generator with optional seed.

    This creates an isolated random generator that doesn't affect
    the global random state.

    Args:
        seed: Optional seed for the generator

    Returns:
        NumPy random generator instance

    Example:
        rng = get_rng(42)
        samples = rng.normal(0, 1, size=100)
    """
    return np.random.default_rng(seed)


def ensure_reproducibility(seed: int = DEFAULT_SEED) -> None:
    """Configure environment for maximum reproducibility.

    This sets seeds and configures various libraries for
    deterministic behavior where possible.

    Args:
        seed: Random seed value

    Note:
        Some operations may still be non-deterministic due to
        floating-point precision and parallel computation.
    """
    set_seeds(seed)

    # Additional PyTorch reproducibility settings
    if PYTORCH_AVAILABLE:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("PyTorch CUDA deterministic mode enabled")

    logger.info("Reproducibility configured with seed: %d", seed)
