"""
Format Handlers for Output Manager

Serialization and deserialization logic for all supported output formats.
Extracted from output_manager.py as part of monolith decomposition (#2486).
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import]

from ..core.datetime_utils import format_datetime, timestamp_iso
from .common_utils import get_logger
from .provenance import ProvenanceInfo, add_provenance_header_file

logger = get_logger(__name__)


class OutputFormat(Enum):
    """Supported output formats."""

    CSV = "csv"
    JSON = "json"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    PARQUET = "parquet"


def _make_json_serializer() -> Any:
    """Return a JSON default serializer for numpy/datetime types."""

    def json_serializer(obj: Any) -> Any:
        """Serialize numpy arrays, numbers, and datetimes to JSON-safe types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer | np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return format_datetime(obj, "iso")
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json_serializer


def save_csv(
    results: pd.DataFrame | dict[str, Any] | list[dict[str, Any]] | np.ndarray,
    file_path: Path,
    provenance: ProvenanceInfo,
) -> None:
    """Save results in CSV format with provenance header."""
    is_df = False
    try:
        if isinstance(results, pd.DataFrame):
            csv_text = results.to_csv(index=False)
            is_df = True
    except TypeError:
        csv_text = ""

    if not is_df:
        df = pd.DataFrame(results)
        csv_text = df.to_csv(index=False)

    with open(file_path, "w") as f:
        add_provenance_header_file(f, provenance)
        f.write(csv_text)


def save_json(
    results: pd.DataFrame | dict[str, Any] | list[dict[str, Any]] | np.ndarray,
    file_path: Path,
    provenance: ProvenanceInfo,
    metadata: dict[str, Any] | None,
    engine: str,
) -> None:
    """Save results in JSON format with provenance and metadata."""
    output_data = {
        "metadata": metadata or {},
        "provenance": {
            "software": f"{provenance.software_name} v{provenance.software_version}",
            "timestamp_utc": provenance.timestamp_utc,
            "git_commit": provenance.git_commit_sha,
            "git_branch": provenance.git_branch,
            "git_dirty": provenance.git_is_dirty,
            "python_version": provenance.python_version,
            "numpy_version": provenance.numpy_version,
        },
        "results": results,
        "timestamp": timestamp_iso(utc=False),
        "engine": engine,
    }

    with open(file_path, "w") as f:
        json.dump(output_data, f, indent=2, default=_make_json_serializer())


def save_hdf5(
    results: pd.DataFrame | dict[str, Any] | list[dict[str, Any]] | np.ndarray,
    file_path: Path,
) -> None:
    """Save results in HDF5 format."""
    if isinstance(results, pd.DataFrame):
        results.to_hdf(file_path, key="data", mode="w")
    else:
        df = pd.DataFrame(results)
        df.to_hdf(file_path, key="data", mode="w")


def save_parquet(
    results: pd.DataFrame | dict[str, Any] | list[dict[str, Any]] | np.ndarray,
    file_path: Path,
) -> None:
    """Save results in Parquet format."""
    if isinstance(results, pd.DataFrame):
        results.to_parquet(file_path, index=False)
    else:
        df = pd.DataFrame(results)
        df.to_parquet(file_path, index=False)


def dispatch_save(
    results: Any,
    file_path: Path,
    format_type: OutputFormat,
    provenance: ProvenanceInfo,
    metadata: Any,
    engine: Any,
) -> None:
    """Route a save operation to the correct format handler."""
    if format_type == OutputFormat.CSV:
        save_csv(results, file_path, provenance)
    elif format_type == OutputFormat.JSON:
        save_json(results, file_path, provenance, metadata, engine)
    elif format_type == OutputFormat.HDF5:
        save_hdf5(results, file_path)
    elif format_type == OutputFormat.PICKLE:
        raise ValueError(
            "Security: Pickle format is disabled due to deserialization risks. "
            "Use JSON or PARQUET."
        )
    elif format_type == OutputFormat.PARQUET:
        save_parquet(results, file_path)


def dispatch_load(
    file_path: Path,
    format_type: OutputFormat,
) -> pd.DataFrame | dict[str, Any] | list[dict[str, Any]]:
    """Route a load operation to the correct format handler."""
    if format_type == OutputFormat.CSV:
        return pd.read_csv(file_path, comment="#")

    if format_type == OutputFormat.JSON:
        with open(file_path) as f:
            data = json.load(f)
        result = data.get("results", data) if isinstance(data, dict) else data
        return dict(result) if isinstance(result, dict) else result

    if format_type == OutputFormat.HDF5:
        result = pd.read_hdf(file_path, key="data")
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"Expected HDF5 key 'data' to contain a pandas DataFrame, "
                f"but got {type(result).__name__}"
            )
        return result

    if format_type == OutputFormat.PICKLE:
        raise ValueError(
            "Security: Pickle format is disabled due to deserialization risks. "
            "Use JSON or PARQUET."
        )

    if format_type == OutputFormat.PARQUET:
        return pd.read_parquet(file_path)

    raise ValueError(f"Unsupported format: {format_type}")
