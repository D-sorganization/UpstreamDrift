"""Backward-compatible output manager module path.

Legacy imports use ``shared.python.output_manager`` while the implementation
now lives in ``shared.python.data_io.output_manager``.
"""

from src.shared.python.data_io.output_manager import (
    OutputFormat,
    OutputManager,
)


def save_results(
    results, filename: str, format_type: str = "csv", engine: str = "mujoco"
) -> str:
    """Backward-compatible convenience save helper."""
    manager = OutputManager()
    return str(
        manager.save_simulation_results(
            results,
            filename,
            OutputFormat(format_type),
            engine,
        )
    )


def load_results(filename: str, format_type: str = "csv", engine: str = "mujoco"):
    """Backward-compatible convenience load helper."""
    manager = OutputManager()
    return manager.load_simulation_results(filename, OutputFormat(format_type), engine)


__all__ = ["OutputFormat", "OutputManager", "save_results", "load_results"]
