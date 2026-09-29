# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Persistence layer for saving/loading solutions and app state.

Handles JSON serialization of OptimizationResult objects, including
numpy array conversion to/from nested Python lists.

Design Principles:
    DBC  -- preconditions checked at function entry.
    DRY  -- array conversion logic is factored into helpers.
    LoD  -- callers pass only the data they want persisted.

Schema validation:
    All persisted JSON files include a ``schema_version`` field. Loading
    a file with a missing/unknown version, missing required keys, wrong
    value types, or out-of-range numeric values raises
    :class:`InvalidStateFileError` (a ``ValueError`` subclass) with a
    message that names the offending field. Issue #403.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_app_paths
from .trajectory import OptimizationResult
from .validation import (
    BAR_MASS_RANGE,
    BODY_MASS_RANGE,
    DURATION_RANGE,
    HEIGHT_RANGE,
    SMOOTHNESS_RANGE,
)

logger = logging.getLogger(__name__)

# Schema version persisted on every save. Bump when the on-disk layout
# changes incompatibly. Loaders reject other versions until a migration
# is implemented.
SCHEMA_VERSION = 1

# Public result-format version for saved solution payloads. This is separate
# from ``SCHEMA_VERSION`` so loaders can reject newer result formats while
# still keeping app-state schema validation stable.
RESULT_FORMAT_VERSION = 2

# Known exercise identifiers accepted by ``save_solution``. Keep in sync
# with ``MainWindow.EXERCISE_CONFIGS`` and the exercise factories.
_KNOWN_EXERCISE_TYPES = frozenset(
    {
        "squat",
        "full_squat",
        "deadlift",
        "bench_press",
        "clean",
        "jerk",
        "snatch",
        "gait",
        "sit_to_stand",
    }
)

# Array field names in OptimizationResult
_ARRAY_FIELDS = ("t", "q", "qd", "qdd", "torques", "power", "com", "bar")

# Scalar metadata fields in OptimizationResult
_METADATA_FIELDS = (
    "success",
    "cost",
    "com_horizontal_range_cm",
    "elapsed_s",
    "n_evals",
)

# Slider keys persisted by ``collect_slider_values`` and the validator
# that constrains each. Mass/height/etc. reuse the bounds in
# ``validation.py`` so the GUI and disk format stay in lockstep.
_SLIDER_RANGES: dict[str, tuple[float, float]] = {
    "body_mass": BODY_MASS_RANGE,
    "height": HEIGHT_RANGE,
    "lower_leg": (0.5, 1.5),  # segment-length multiplier
    "upper_leg": (0.5, 1.5),
    "torso": (0.5, 1.5),
    "bar_mass": BAR_MASS_RANGE,
    "duration": DURATION_RANGE,
    "smoothness": SMOOTHNESS_RANGE,
}


class InvalidStateFileError(ValueError):
    """Raised when a persisted JSON file fails schema validation.

    Subclasses :class:`ValueError` so existing callers that already
    catch ``ValueError`` (e.g. the GUI load path) keep working without
    code changes.
    """


# ---------------------------------------------------------------------------
# Schema validation helpers
# ---------------------------------------------------------------------------


def _require_mapping(data: Any, context: str) -> dict[str, Any]:
    """Return ``data`` as a dict or raise with a descriptive context."""
    if not isinstance(data, dict):
        raise InvalidStateFileError(f"{context}: expected JSON object, got {type(data).__name__}")
    return data


def _require_key(data: dict[str, Any], key: str, context: str) -> Any:
    if key not in data:
        raise InvalidStateFileError(f"{context}: missing required key '{key}'")
    return data[key]


def _require_type(value: Any, expected: type | tuple[type, ...], field: str) -> None:
    # ``bool`` is a subclass of ``int`` in Python; reject it explicitly when
    # a numeric value is expected so True/False can't sneak in as 1/0.
    if isinstance(expected, tuple):
        numeric_expected = any(t in (int, float) for t in expected)
    else:
        numeric_expected = expected in (int, float)
    if numeric_expected and isinstance(value, bool):
        raise InvalidStateFileError(f"field '{field}': expected number, got bool")
    if not isinstance(value, expected):
        names = (
            expected.__name__
            if isinstance(expected, type)
            else "/".join(t.__name__ for t in expected)
        )
        raise InvalidStateFileError(
            f"field '{field}': expected {names}, got {type(value).__name__}"
        )


def _require_range(value: float, bounds: tuple[float, float], field: str) -> None:
    low, high = bounds
    if not (low <= value <= high):
        raise InvalidStateFileError(f"field '{field}': value {value} out of range [{low}, {high}]")


def _validate_schema_version(data: dict[str, Any], context: str) -> None:
    """Reject files with a missing or unsupported schema version."""
    if "schema_version" not in data:
        raise InvalidStateFileError(
            f"{context}: missing 'schema_version' (file predates schema "
            f"validation; expected version {SCHEMA_VERSION})"
        )
    version = data["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise InvalidStateFileError(
            f"{context}: 'schema_version' must be int, got {type(version).__name__}"
        )
    if version != SCHEMA_VERSION:
        raise InvalidStateFileError(
            f"{context}: unsupported schema_version {version} "
            f"(expected {SCHEMA_VERSION}); no migration is available"
        )


def _validate_arrays_block(arrays: Any, context: str) -> None:
    arrays_dict = _require_mapping(arrays, f"{context}: 'arrays'")
    for key in _ARRAY_FIELDS:
        if key not in arrays_dict:
            raise InvalidStateFileError(f"{context}: missing required array '{key}'")
        value = arrays_dict[key]
        if not isinstance(value, list):
            raise InvalidStateFileError(
                f"{context}: 'arrays.{key}' must be a list, got {type(value).__name__}"
            )


def _validate_metadata_block(metadata: Any, context: str) -> None:
    meta_dict = _require_mapping(metadata, f"{context}: 'metadata'")
    required_types: dict[str, type | tuple[type, ...]] = {
        "success": bool,
        "cost": (int, float),
        "com_horizontal_range_cm": (int, float),
    }
    for key, expected in required_types.items():
        if key not in meta_dict:
            raise InvalidStateFileError(f"{context}: missing required metadata key '{key}'")
        # ``success`` is bool and must be checked separately to avoid the
        # numeric-bool guard in ``_require_type``.
        if expected is bool:
            if not isinstance(meta_dict[key], bool):
                raise InvalidStateFileError(
                    f"field 'metadata.{key}': expected bool, got {type(meta_dict[key]).__name__}"
                )
        else:
            _require_type(meta_dict[key], expected, f"metadata.{key}")
    # Optional numeric fields with defaults.
    for opt_key in ("elapsed_s", "n_evals", "n_joint_limit_violations"):
        if opt_key in meta_dict:
            _require_type(meta_dict[opt_key], (int, float), f"metadata.{opt_key}")


def _validate_solution_schema(data: dict[str, Any]) -> None:
    """Validate the on-disk shape of a saved solution file."""
    context = "solution file"
    _validate_schema_version(data, context)

    format_version = data.get("format_version", 1)
    if not isinstance(format_version, int) or isinstance(format_version, bool):
        raise InvalidStateFileError(
            f"{context}: 'format_version' must be int, got {type(format_version).__name__}"
        )
    if format_version > RESULT_FORMAT_VERSION:
        raise InvalidStateFileError(
            f"{context}: unsupported format_version {format_version} "
            f"(expected <= {RESULT_FORMAT_VERSION})"
        )

    exercise_type = _require_key(data, "exercise_type", context)
    _require_type(exercise_type, str, "exercise_type")
    if exercise_type not in _KNOWN_EXERCISE_TYPES:
        raise InvalidStateFileError(
            f"field 'exercise_type': unknown value '{exercise_type}' "
            f"(known: {sorted(_KNOWN_EXERCISE_TYPES)})"
        )

    bar_mass = _require_key(data, "bar_mass", context)
    _require_type(bar_mass, (int, float), "bar_mass")
    _require_range(float(bar_mass), BAR_MASS_RANGE, "bar_mass")

    body_params = _require_key(data, "body_params", context)
    _require_type(body_params, dict, "body_params")

    arrays = _require_key(data, "arrays", context)
    _validate_arrays_block(arrays, context)
    metadata = _require_key(data, "metadata", context)
    _validate_metadata_block(metadata, context)


def _validate_app_state_schema(data: dict[str, Any]) -> None:
    """Validate the on-disk shape of a saved app-state file."""
    context = "app state file"
    _validate_schema_version(data, context)

    slider_values = _require_key(data, "slider_values", context)
    _require_type(slider_values, dict, "slider_values")
    for key, value in slider_values.items():
        if not isinstance(key, str):
            raise InvalidStateFileError(
                f"slider_values: keys must be strings, got {type(key).__name__}"
            )
        _require_type(value, (int, float), f"slider_values.{key}")
        if key in _SLIDER_RANGES:
            _require_range(float(value), _SLIDER_RANGES[key], f"slider_values.{key}")

    results = _require_key(data, "results", context)
    _require_type(results, dict, "results")
    for etype, payload in results.items():
        if not isinstance(etype, str):
            raise InvalidStateFileError(
                f"results: exercise keys must be strings, got {type(etype).__name__}"
            )
        if etype not in _KNOWN_EXERCISE_TYPES:
            raise InvalidStateFileError(
                f"results: unknown exercise type '{etype}' (known: {sorted(_KNOWN_EXERCISE_TYPES)})"
            )
        sub = _require_mapping(payload, f"results.{etype}")
        if "arrays" not in sub or "metadata" not in sub:
            raise InvalidStateFileError(f"results.{etype}: must contain 'arrays' and 'metadata'")
        _validate_arrays_block(sub["arrays"], f"results.{etype}")
        _validate_metadata_block(sub["metadata"], f"results.{etype}")


# ---------------------------------------------------------------------------
# Result <-> dict helpers
# ---------------------------------------------------------------------------


def _result_to_dict(result: OptimizationResult) -> dict[str, Any]:
    """Convert an OptimizationResult to a JSON-serializable dict."""
    arrays = {k: getattr(result, k).tolist() for k in _ARRAY_FIELDS}
    metadata = {k: getattr(result, k) for k in _METADATA_FIELDS}
    return {"arrays": arrays, "metadata": metadata}


def _dict_to_result(d: dict[str, Any]) -> OptimizationResult:
    """Reconstruct an OptimizationResult from a dict."""
    arrays = {k: np.array(d["arrays"][k]) for k in _ARRAY_FIELDS}
    meta = d["metadata"]
    return OptimizationResult(
        t=arrays["t"],
        q=arrays["q"],
        qd=arrays["qd"],
        qdd=arrays["qdd"],
        torques=arrays["torques"],
        power=arrays["power"],
        com=arrays["com"],
        bar=arrays["bar"],
        success=meta["success"],
        cost=meta["cost"],
        com_horizontal_range_cm=meta["com_horizontal_range_cm"],
        elapsed_s=meta.get("elapsed_s", 0.0),
        n_evals=meta.get("n_evals", 0),
        n_joint_limit_violations=meta.get("n_joint_limit_violations", 0),
    )


def solution_data_to_result(data: dict[str, Any]) -> OptimizationResult:
    """Reconstruct an :class:`OptimizationResult` from loaded solution data.

    Args:
        data: Solution payload returned by :func:`load_solution` or an
            equivalent dict with ``arrays`` and ``metadata`` blocks.

    Returns:
        The reconstructed optimization result.

    Raises:
        InvalidStateFileError: If a full solution payload fails schema
            validation.
    """
    if "schema_version" in data:
        _validate_solution_schema(data)
    return _dict_to_result(data)


# ---------------------------------------------------------------------------
# Public save/load API
# ---------------------------------------------------------------------------


def save_solution(
    path: str | Path,
    result: OptimizationResult,
    body_params: dict[str, Any],
    exercise_type: str,
    bar_mass: float,
) -> None:
    """Save an optimization solution to a JSON file.

    Preconditions:
        path is a writable file path.
        result is a valid OptimizationResult.
    """
    from . import __version__

    data = {
        "schema_version": SCHEMA_VERSION,
        "format_version": RESULT_FORMAT_VERSION,
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "export_app_version": __version__,
        "exercise_type": exercise_type,
        "bar_mass": bar_mass,
        "body_params": body_params,
        **_result_to_dict(result),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved solution to %s", output_path)


def load_solution(path: str | Path) -> dict[str, Any]:
    """Load an optimization solution from a JSON file.

    Returns a dict with keys: schema_version, exercise_type, bar_mass,
    body_params, arrays (dict of list data), metadata (dict of scalars).

    Raises:
        FileNotFoundError: if path does not exist.
        json.JSONDecodeError: if file is not valid JSON.
        InvalidStateFileError: if the file fails schema validation.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Solution file not found: {input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise InvalidStateFileError(
            f"solution file: expected JSON object at top level, got {type(data).__name__}"
        )
    _validate_solution_schema(data)
    logger.info("Loaded solution from %s", input_path)
    return data


def save_app_state(
    results_dict: dict[str, OptimizationResult],
    slider_values: dict[str, float],
    *,
    state_dir: str | Path | None = None,
) -> None:
    """Save the current app state to the state directory.

    Preconditions:
        results_dict maps exercise_type -> OptimizationResult.
        slider_values maps slider_name -> float value.
    """
    state_path = (
        load_app_paths().state_file if state_dir is None else Path(state_dir) / "last_state.json"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)

    serialized_results = {}
    for etype, result in results_dict.items():
        if result is not None:
            serialized_results[etype] = _result_to_dict(result)

    data = {
        "schema_version": SCHEMA_VERSION,
        "slider_values": slider_values,
        "results": serialized_results,
    }
    state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved app state to %s", state_path)


def load_app_state(*, state_dir: str | Path | None = None) -> dict[str, Any] | None:
    """Load the last app state from the state directory.

    Returns None if no state file exists or if the file is not valid
    JSON (corruption that predates schema validation). Raises
    :class:`InvalidStateFileError` when the file parses but fails
    schema validation -- the GUI surfaces this so the user knows their
    state is incompatible rather than being silently discarded.
    """
    state_path = (
        load_app_paths().state_file if state_dir is None else Path(state_dir) / "last_state.json"
    )

    if not state_path.exists():
        return None

    try:
        raw = state_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read app state file %s: %s", state_path, exc)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("App state file %s is not valid JSON: %s", state_path, exc)
        return None

    if not isinstance(data, dict):
        raise InvalidStateFileError(
            f"app state file: expected JSON object at top level, got {type(data).__name__}"
        )
    _validate_app_state_schema(data)
    logger.info("Loaded app state from %s", state_path)
    return data
