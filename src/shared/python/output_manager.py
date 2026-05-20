"""Backward-compatible output manager module path.

Legacy imports use ``shared.python.output_manager`` while the implementation
now lives in ``shared.python.data_io.output_manager``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.shared.python.data_io.output_manager import (
    OutputFormat,
    OutputManager,
)


def save_results(
    results: Any, filename: str, format_type: str = "csv", engine: str = "mujoco"
) -> str:
    """Backward-compatible convenience save helper."""
    if results is None:
        raise ValueError("results must be provided")
    manager = OutputManager()
    return str(
        manager.save_simulation_results(
            results,
            filename,
            OutputFormat(format_type),
            engine,
        )
    )


def load_results(
    filename: str, format_type: str = "csv", engine: str = "mujoco"
) -> pd.DataFrame | dict[str, Any] | list[dict[str, Any]]:
    """Backward-compatible convenience load helper."""
    if filename is None:
        raise ValueError("filename must be provided")
    manager = OutputManager()
    return manager.load_simulation_results(filename, OutputFormat(format_type), engine)


__all__ = ["OutputFormat", "OutputManager", "save_results", "load_results"]
