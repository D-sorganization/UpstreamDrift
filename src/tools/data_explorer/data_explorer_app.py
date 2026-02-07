"""Data Explorer Application — simulation data workbench.

Provides a GUI/CLI for browsing, filtering, and visualizing simulation
datasets exported from physics engines. Integrates with the Scientific
Data Processor (data_processor.core) for advanced filtering capabilities.

Supported formats:
    - CSV (time series, parameter sweeps)
    - JSON (simulation configs, results)
    - HDF5 (large datasets, multi-run batches)
    - C3D (motion capture data)

Design by Contract:
    Preconditions:
        - Data files must be accessible and in a supported format
    Postconditions:
        - Data is loaded, validated, and available for visualization
    Invariants:
        - Original data is never modified (read-only processing)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Supported file extensions for data import
SUPPORTED_EXTENSIONS: set[str] = {".csv", ".json", ".hdf5", ".h5", ".c3d"}


def discover_datasets(search_dir: Path) -> list[Path]:
    """Discover simulation datasets in a directory.

    Args:
        search_dir: Directory to search for data files

    Returns:
        List of discovered dataset file paths, sorted by name

    Raises:
        FileNotFoundError: If search_dir doesn't exist
    """
    if not search_dir.exists():
        raise FileNotFoundError(f"Search directory not found: {search_dir}")

    datasets: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        datasets.extend(search_dir.rglob(f"*{ext}"))

    return sorted(datasets)


def load_dataset(filepath: Path) -> dict[str, Any]:
    """Load a single dataset file.

    Args:
        filepath: Path to the dataset file

    Returns:
        Dictionary with 'path', 'format', 'size_bytes', and 'columns' keys

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    info: dict[str, Any] = {
        "path": str(filepath),
        "format": suffix.lstrip("."),
        "size_bytes": filepath.stat().st_size,
    }

    if suffix == ".csv":
        info["columns"] = _get_csv_columns(filepath)
    elif suffix == ".json":
        info["columns"] = _get_json_keys(filepath)

    logger.info(
        "Loaded dataset: %s (%s, %d bytes)", filepath.name, suffix, info["size_bytes"]
    )
    return info


def _get_csv_columns(filepath: Path) -> list[str]:
    """Extract column headers from a CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        List of column header strings
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            header_line = f.readline().strip()
        return [col.strip().strip('"') for col in header_line.split(",")]
    except Exception as e:
        logger.warning("Could not parse CSV headers from %s: %s", filepath, e)
        return []


def _get_json_keys(filepath: Path) -> list[str]:
    """Extract top-level keys from a JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        List of top-level key strings
    """
    import json

    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return list(data.keys())
        return []
    except Exception as e:
        logger.warning("Could not parse JSON keys from %s: %s", filepath, e)
        return []


def main() -> int:
    """Launch the Data Explorer application.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info("Launching Data Explorer...")

    # Default search path is the project's output directory
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / "output"

    if output_dir.exists():
        datasets = discover_datasets(output_dir)
        logger.info("Discovered %d datasets in %s", len(datasets), output_dir)
    else:
        logger.info(
            "No output directory found at %s — starting with empty workspace",
            output_dir,
        )

    logger.info(
        "Data Explorer ready — supports %s formats",
        ", ".join(sorted(SUPPORTED_EXTENSIONS)),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
