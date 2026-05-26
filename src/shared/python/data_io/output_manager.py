# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Output Manager for Golf Modeling Suite

Handles all output operations including saving simulation results,
managing file organization, and exporting analysis reports.

OBS-001: Migrated to structured logging for better observability.

PERFORMANCE: Added async file I/O support using ThreadPoolExecutor
to prevent blocking the main thread during large file writes.

Implementation is split across submodules for maintainability (#2486):
  _format_handlers.py  — CSV/JSON/HDF5/Parquet save & load dispatch
  _path_utils.py       — filename sanitization, dir scanning, project-root resolution
  _async_io.py         — ThreadPoolExecutor-based background saves
  _report_generators.py — analysis report export (JSON, HTML)
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd  # type: ignore[import]

from ..core.contracts import invariant, precondition

# Re-imported for backward compatibility with tests that patch this name here
from ..core.datetime_utils import now_local  # noqa: F401
from ._async_io import (
    get_io_executor,
    shutdown_executor,
    submit_async_save,
    submit_background_save,
)

# Submodule imports — re-exported for backward compatibility
from ._format_handlers import OutputFormat, dispatch_load, dispatch_save
from ._path_utils import (
    create_output_structure,
    fast_dir_scan,
    resolve_base_path,
    sanitize_filename,
)
from ._report_generators import export_analysis_report
from ._simulation_store import cleanup_old_files, get_simulation_list
from .common_utils import get_logger, setup_structured_logging
from .provenance import ProvenanceInfo

# Configure structured logging
setup_structured_logging()
logger = get_logger(__name__)


@invariant(
    lambda self: self.base_path.exists(),
    "Output base_path directory must exist",
)
class OutputManager:
    """
    Manages all output operations for the Golf Modeling Suite.

    Provides unified interface for saving simulation results, analysis data,
    and generating reports across all physics engines.

    PERFORMANCE: Includes async file I/O support to prevent blocking main thread.
    """

    @classmethod
    def _get_io_executor(cls):
        """Get or create the shared I/O executor."""
        return get_io_executor()

    @classmethod
    def shutdown_executor(cls) -> None:
        """Shutdown the I/O executor. Call on application exit."""
        shutdown_executor()

    def __init__(self, base_path: str | Path | None = None) -> None:
        """
        Initialize OutputManager.

        Args:
            base_path: Base directory for outputs. Defaults to 'output' in project root.
        """
        self.base_path = resolve_base_path(base_path)

        # Define standard subdirectories
        self.directories = {
            "simulations": self.base_path / "simulations",
            "analysis": self.base_path / "analysis",
            "exports": self.base_path / "exports",
            "reports": self.base_path / "reports",
            "cache": self.base_path / "cache",
        }

        logger.info(
            "output_manager_initialized base_path=%s num_directories=%d",
            self.base_path,
            len(self.directories),
        )

    def create_output_structure(self) -> None:
        """Create the standard output directory structure."""
        create_output_structure(self.directories)

    @precondition(  # fmt: skip
        lambda self, results, filename, format_type=OutputFormat.CSV, engine="mujoco", metadata=None, model_path=None, parameters=None: (
            results is not None
        ),
        "Simulation results must not be None",
    )
    @precondition(  # fmt: skip
        lambda self, results, filename, format_type=OutputFormat.CSV, engine="mujoco", metadata=None, model_path=None, parameters=None: (
            filename is not None and len(filename) > 0
        ),
        "Filename must be a non-empty string",
    )
    def save_simulation_results(
        self,
        results: pd.DataFrame | dict[str, Any] | list[dict[str, Any]] | np.ndarray,
        filename: str,
        format_type: OutputFormat = OutputFormat.CSV,
        engine: str = "mujoco",
        metadata: dict[str, Any] | None = None,
        model_path: Path | str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save simulation results to file.

        Args:
            results: Simulation results data
            filename: Output filename (without extension)
            format_type: Output format
            engine: Physics engine name
            metadata: Additional metadata to include
            model_path: Optional path to the model file (for provenance)
            parameters: Optional simulation parameters (for provenance)

        Returns:
            Path to saved file
        """
        if results is None:
            raise ValueError("results must be provided")
        engine_dir = self.directories["simulations"] / engine
        engine_dir.mkdir(parents=True, exist_ok=True)

        clean_filename = sanitize_filename(filename, format_type)
        file_path = engine_dir / f"{clean_filename}.{format_type.value}"

        provenance = ProvenanceInfo.capture(
            model_path=model_path, parameters=parameters
        )

        try:
            dispatch_save(results, file_path, format_type, provenance, metadata, engine)

            logger.info(
                "simulation_results_saved file_path=%s format=%s engine=%s",
                file_path,
                format_type.value,
                engine,
            )
            return file_path

        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(
                "simulation_save_failed filename=%s format=%s engine=%s error=%s",
                filename,
                format_type.value,
                engine,
                e,
                exc_info=True,
            )
            raise

    def save_simulation_results_async(
        self,
        results: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
        filename: str,
        format_type: OutputFormat = OutputFormat.CSV,
        engine: str = "mujoco",
        metadata: dict[str, Any] | None = None,
        callback: Callable[[Path | Exception], None] | None = None,
    ) -> Future[Path]:
        """
        Save simulation results asynchronously without blocking.

        PERFORMANCE: Uses ThreadPoolExecutor to perform I/O in background.
        Ideal for large files where blocking would impact simulation performance.

        Args:
            results: Simulation results data
            filename: Output filename (without extension)
            format_type: Output format
            engine: Physics engine name
            metadata: Additional metadata to include
            callback: Optional callback called with Path on success or Exception on failure

        Returns:
            Future that resolves to the saved file path
        """
        if results is None:
            raise ValueError("results must be provided")
        return submit_async_save(
            self.save_simulation_results,
            results,
            filename,
            format_type,
            engine,
            metadata,
            callback=callback,
        )

    def save_simulation_results_background(
        self,
        results: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
        filename: str,
        format_type: OutputFormat = OutputFormat.CSV,
        engine: str = "mujoco",
        metadata: dict[str, Any] | None = None,
        on_complete: Callable[[Path], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """
        Fire-and-forget background save with optional callbacks.

        PERFORMANCE: For cases where you don't need the result immediately.
        Useful for auto-save, checkpointing, or progress exports.

        Args:
            results: Simulation results data
            filename: Output filename (without extension)
            format_type: Output format
            engine: Physics engine name
            metadata: Additional metadata to include
            on_complete: Called with file path on successful save
            on_error: Called with exception on failure
        """
        if results is None:
            raise ValueError("results must be provided")
        submit_background_save(
            self.save_simulation_results,
            results,
            filename,
            format_type,
            engine,
            metadata,
            on_complete=on_complete,
            on_error=on_error,
        )

    def load_simulation_results(
        self,
        filename: str,
        format_type: OutputFormat = OutputFormat.CSV,
        engine: str = "mujoco",
    ) -> pd.DataFrame | dict[str, Any] | list[dict[str, Any]]:
        """
        Load simulation results from file.

        Args:
            filename: Input filename
            format_type: File format
            engine: Physics engine name

        Returns:
            Loaded simulation results
        """
        engine_dir = self.directories["simulations"] / engine

        if not filename.endswith(f".{format_type.value}"):
            filename = f"{filename}.{format_type.value}"

        file_path = engine_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Simulation file not found: {file_path}")

        try:
            return dispatch_load(file_path, format_type)
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error("Error loading simulation results: %s", e)
            raise

    def get_simulation_list(self, engine: str | None = None) -> list[str]:
        """
        Get list of available simulation files.

        Args:
            engine: Filter by specific engine (optional)

        Returns:
            List of simulation filenames
        """
        return get_simulation_list(self.directories["simulations"], engine)

    @precondition(
        lambda self, analysis_data, report_name, format_type="json": (
            analysis_data is not None
        ),
        "Analysis data must not be None",
    )
    @precondition(
        lambda self, analysis_data, report_name, format_type="json": (
            report_name is not None and len(report_name) > 0
        ),
        "Report name must be a non-empty string",
    )
    def export_analysis_report(
        self,
        analysis_data: dict[str, Any],
        report_name: str,
        format_type: str = "json",
    ) -> Path:
        """
        Export analysis report.

        Args:
            analysis_data: Analysis results and metadata
            report_name: Report filename (without extension)
            format_type: Report format (json, html, pdf)

        Returns:
            Path to exported report
        """
        report_dir = self.directories["reports"] / format_type
        return export_analysis_report(
            analysis_data, report_name, report_dir, format_type
        )

    @precondition(
        lambda self, max_age_days=30: max_age_days > 0,
        "Maximum age in days must be positive",
    )
    def cleanup_old_files(self, max_age_days: int = 30) -> int:
        """
        Clean up old files based on age.

        Args:
            max_age_days: Maximum age in days before cleanup

        Returns:
            Number of files cleaned up
        """
        return cleanup_old_files(
            cache_dir=self.directories["cache"],
            simulations_dir=self.directories["simulations"],
            analysis_dir=self.directories["analysis"],
            base_path=self.base_path,
            max_age_days=max_age_days,
        )

    # Kept for backward compatibility — thin delegation to _path_utils
    @staticmethod
    def _fast_dir_scan(directory: Path, max_depth: int = 10):
        """Fast directory scanning. Delegates to _path_utils.fast_dir_scan."""
        return fast_dir_scan(directory, max_depth)


# ---------------------------------------------------------------------------
# Type aliases (TYPE-001: Improved type safety over Any)
# ---------------------------------------------------------------------------

SimulationResultScalar: TypeAlias = int | float | str | bool | None
# Helper alias for recursive dict
SimulationResultValue: TypeAlias = (
    SimulationResultScalar
    | list[SimulationResultScalar]
    | dict[str, SimulationResultScalar]
    # Allow simple nested lists of dictionaries for JSON-like structures
    | list[dict[str, Any]]
)
# Using Any for dict values to handle arbitrarily nested JSON structures better than
# recursive types which mypy struggles with in some contexts.
SimulationResultDict: TypeAlias = dict[str, Any]

SimulationResults: TypeAlias = (
    SimulationResultDict | pd.DataFrame | np.ndarray | list[dict[str, Any]]
)


# ---------------------------------------------------------------------------
# Convenience functions for backward compatibility
# ---------------------------------------------------------------------------


def save_results(
    results: SimulationResults,
    filename: str,
    format_type: str = "csv",
    engine: str = "mujoco",
) -> str:
    """Convenience function for saving results.

    TYPE-001: Replaced Any with Union type for better type safety.
    """
    if results is None:
        raise ValueError("results must be provided")
    manager = OutputManager()
    path = manager.save_simulation_results(
        results,
        filename,
        OutputFormat(format_type),
        engine,  # type: ignore[arg-type]
    )
    return str(path)


def load_results(
    filename: str, format_type: str = "csv", engine: str = "mujoco"
) -> SimulationResults:
    """Convenience function for loading results.

    TYPE-001: Replaced Any with Union type for better type safety.
    """
    if filename is None:
        raise ValueError("filename must be provided")
    manager = OutputManager()
    result = manager.load_simulation_results(
        filename, OutputFormat(format_type), engine
    )
    return result
