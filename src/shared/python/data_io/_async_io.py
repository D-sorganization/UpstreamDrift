"""
Async I/O support for Output Manager

ThreadPoolExecutor-based background save operations to prevent blocking
the main simulation thread during large file writes.
Extracted from output_manager.py as part of monolith decomposition (#2486).

PERFORMANCE: Added async file I/O support using ThreadPoolExecutor
to prevent blocking the main thread during large file writes.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import]

from ._format_handlers import OutputFormat
from .common_utils import get_logger

logger = get_logger(__name__)

# PERFORMANCE: Shared executor for async I/O operations
_io_executor: ThreadPoolExecutor | None = None
_MAX_IO_WORKERS = 4  # Limit concurrent I/O operations


def get_io_executor() -> ThreadPoolExecutor:
    """Get or create the shared I/O executor."""
    global _io_executor
    if _io_executor is None:
        _io_executor = ThreadPoolExecutor(
            max_workers=_MAX_IO_WORKERS, thread_name_prefix="output_io"
        )
    return _io_executor


def shutdown_executor() -> None:
    """Shutdown the I/O executor. Call on application exit."""
    global _io_executor
    if _io_executor is not None:
        _io_executor.shutdown(wait=True)
        _io_executor = None


def submit_async_save(
    save_fn: Callable[..., Path],
    results: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    filename: str,
    format_type: OutputFormat,
    engine: str,
    metadata: dict[str, Any] | None,
    callback: Callable[[Path | Exception], None] | None = None,
) -> Future[Path]:
    """
    Submit a save operation to the background executor.

    PERFORMANCE: Uses ThreadPoolExecutor to perform I/O in background.
    Ideal for large files where blocking would impact simulation performance.

    Args:
        save_fn: The synchronous save callable (e.g. OutputManager.save_simulation_results).
        results: Simulation results data.
        filename: Output filename (without extension).
        format_type: Output format.
        engine: Physics engine name.
        metadata: Additional metadata to include.
        callback: Optional callback called with Path on success or Exception on failure.

    Returns:
        Future that resolves to the saved file path.
    """
    executor = get_io_executor()

    def _save_task() -> Path:
        try:
            path = save_fn(results, filename, format_type, engine, metadata)
            if callback:
                callback(path)
            return path
        except (RuntimeError, TypeError, ValueError) as e:
            if callback:
                callback(e)
            raise

    future = executor.submit(_save_task)
    logger.debug(
        "async_save_submitted filename=%s format=%s engine=%s",
        filename,
        format_type.value,
        engine,
    )
    return future


def submit_background_save(
    save_fn: Callable[..., Path],
    results: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    filename: str,
    format_type: OutputFormat,
    engine: str,
    metadata: dict[str, Any] | None,
    on_complete: Callable[[Path], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    """
    Fire-and-forget background save with optional callbacks.

    PERFORMANCE: For cases where you don't need the result immediately.
    Useful for auto-save, checkpointing, or progress exports.

    Args:
        save_fn: The synchronous save callable.
        results: Simulation results data.
        filename: Output filename (without extension).
        format_type: Output format.
        engine: Physics engine name.
        metadata: Additional metadata to include.
        on_complete: Called with file path on successful save.
        on_error: Called with exception on failure.
    """

    def _callback(result: Path | Exception) -> None:
        if isinstance(result, Exception):
            if on_error:
                on_error(result)
            else:
                logger.error(
                    "background_save_failed filename=%s error=%s",
                    filename,
                    result,
                )
        elif on_complete:
            on_complete(result)

    submit_async_save(
        save_fn, results, filename, format_type, engine, metadata, callback=_callback
    )
