# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Import helpers for loading previously exported optimization results.

Design Principles:
    DBC  -- preconditions checked at function entry; ValueError on violation.
    Logging not print -- uses logging.getLogger(__name__) throughout.
    LoD  -- callers pass only a path; internal JSON parsing is encapsulated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .persistence import load_solution, solution_data_to_result
from .trajectory import OptimizationResult

logger = logging.getLogger(__name__)

EXPORT_FORMAT_VERSION = "1.0"


def import_result_from_json(path: str | Path) -> dict:
    """Load an optimization result from a JSON file.

    The file should have been created by :func:`movement_optimizer.export.export_result_json`.
    Files that contain a ``format_version`` field are checked for compatibility.
    Legacy files that pre-date versioning are accepted with a warning.

    Args:
        path: Path to the JSON result file.

    Returns:
        A dict containing the result data as stored in the file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not a valid JSON object or contains an
            incompatible ``format_version``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in result file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Invalid result file: expected JSON object, got {type(data).__name__}")

    version = data.get("format_version")
    if version is None:
        # Legacy file without version -- try to load with a warning.
        logger.warning("Result file %s has no format_version; attempting legacy load", path)
    elif version != EXPORT_FORMAT_VERSION:
        raise ValueError(
            f"Incompatible format_version '{version}' in {path}; expected '{EXPORT_FORMAT_VERSION}'"
        )

    logger.info("Loaded result from %s (format_version=%s)", path, version)
    return data


def import_results_from_json(path: str | Path) -> OptimizationResult:
    """Import a saved solution JSON file as an ``OptimizationResult``.

    This is the structured round-trip companion to
    :func:`movement_optimizer.persistence.save_solution`. It validates the
    saved solution schema, checks result-format compatibility, and reconstructs
    NumPy arrays from the JSON payload.

    Args:
        path: Path to a JSON solution file.

    Returns:
        The reconstructed optimization result.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        InvalidStateFileError: If the solution schema or format version is
            unsupported.
    """
    data = load_solution(path)
    result = solution_data_to_result(data)
    logger.info("Imported OptimizationResult from %s", path)
    return result
