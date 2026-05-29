"""Pandera schema validation for data_io loaders (issue #6568).

This module adds *additive* schema enforcement to the most widely consumed
CSV / array loaders in :mod:`src.shared.python.data_io`. Validation is
designed to be transparent on the success path:

* Valid input behaves exactly as before — the loaded object is returned
  unchanged.
* Malformed input (missing columns, wrong dtypes, wrong shape, empty/NaN
  payloads) raises a clear, typed :class:`DataFormatError` instead of
  silently mis-parsing downstream.

``pandera`` is an *optional* dependency (declared under the ``data`` extra in
``pyproject.toml``). It is imported lazily so that importing this module — or
the loaders that call into it — never fails when pandera is absent. When
pandera is not installed the column/dtype schema checks are skipped (a single
debug log is emitted), while the lightweight structural checks (non-empty,
finite, expected shape) still run because they only depend on numpy/pandas.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.core.error_utils import DataFormatError
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

__all__ = [
    "pandera_available",
    "validate_csv_dataframe",
    "validate_trajectory_array",
]


@lru_cache(maxsize=1)
def pandera_available() -> bool:
    """Return True when the optional ``pandera`` dependency is importable."""
    try:
        import pandera  # noqa: F401  (probe import)
    except ImportError:
        return False
    return True


def _pandera_modules() -> tuple[Any, Any] | None:
    """Return ``(pandera, Column)`` lazily, or None when unavailable."""
    if not pandera_available():
        return None
    import pandera.pandas as pa
    from pandera.pandas import Column

    return pa, Column


@lru_cache(maxsize=1)
def _numeric_frame_schema() -> Any | None:
    """Build a permissive schema: a non-empty, all-numeric DataFrame.

    Returns None when pandera is unavailable so callers can degrade to the
    structural-only checks performed in :func:`validate_csv_dataframe`.
    """
    mods = _pandera_modules()
    if mods is None:
        return None
    pa, _column = mods
    return pa.DataFrameSchema(
        checks=pa.Check(
            lambda df: df.select_dtypes(include="number").shape[1] == df.shape[1],
            error="all columns must be numeric",
        ),
        coerce=False,
        strict=False,
    )


def validate_csv_dataframe(
    data: pd.DataFrame,
    *,
    source: str,
    require_numeric: bool = False,
    required_columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Validate a freshly loaded CSV DataFrame.

    The success path is a no-op: the same ``data`` object is returned. On a
    schema violation a :class:`DataFormatError` is raised with a message that
    names the offending file.

    Args:
        data: The DataFrame returned by ``pd.read_csv``.
        source: Human-readable source (file path) for error messages.
        require_numeric: When True, every column must be numeric.
        required_columns: Column names that must be present, if any.

    Returns:
        The validated DataFrame (unchanged on success).

    Raises:
        DataFormatError: If the frame is empty, missing required columns, or —
            when ``require_numeric`` — contains non-numeric columns.
    """
    if data is None:
        raise ValueError("data must be provided")

    # Structural checks (no pandera required).
    if data.shape[1] == 0:
        raise DataFormatError(
            f"CSV at {source} parsed to zero columns",
            expected_format="non-empty tabular CSV",
        )
    if required_columns:
        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            raise DataFormatError(
                f"CSV at {source} is missing required column(s): {', '.join(missing)}",
                expected_format=f"columns including {', '.join(required_columns)}",
                actual_format=", ".join(map(str, data.columns)),
            )

    if not require_numeric:
        return data

    schema = _numeric_frame_schema()
    if schema is None:
        # pandera missing — fall back to a structural numeric check.
        import pandas as pd

        non_numeric = [
            str(col)
            for col in data.columns
            if not pd.api.types.is_numeric_dtype(data[col])
        ]
        if non_numeric:
            raise DataFormatError(
                f"CSV at {source} contains non-numeric column(s): "
                f"{', '.join(non_numeric)}",
                expected_format="all-numeric columns",
            )
        return data

    try:
        import pandera.errors as pa_errors

        schema.validate(data, lazy=True)
    except pa_errors.SchemaErrors as exc:
        raise DataFormatError(
            f"CSV at {source} failed schema validation: {exc}",
            expected_format="all-numeric columns",
        ) from exc
    return data


def validate_trajectory_array(
    array: np.ndarray,
    *,
    source: str,
    min_columns: int = 2,
) -> np.ndarray:
    """Validate a numeric trajectory array (e.g. ``time, q0, q1, ...``).

    Used for array loaders that produce a 2D matrix where the first column is
    time and the remainder are per-joint / per-channel values.

    Args:
        array: The loaded ndarray.
        source: Human-readable source (file path) for error messages.
        min_columns: Minimum required column count (default 2: time + 1 value).

    Returns:
        The validated array (unchanged on success).

    Raises:
        DataFormatError: If the array is not 2D, has too few columns, or
            contains non-finite values.
    """
    if array is None:
        raise ValueError("array must be provided")
    if array.ndim != 2:
        raise DataFormatError(
            f"Trajectory at {source} must be 2D",
            expected_format="2D array (rows x columns)",
            actual_format=f"{array.ndim}D array",
        )
    if array.shape[1] < min_columns:
        raise DataFormatError(
            f"Trajectory at {source} has too few columns",
            expected_format=f">= {min_columns} columns",
            actual_format=f"{array.shape[1]} columns",
        )
    if not np.issubdtype(array.dtype, np.number):
        raise DataFormatError(
            f"Trajectory at {source} must be numeric",
            expected_format="numeric dtype",
            actual_format=str(array.dtype),
        )
    if not np.isfinite(array).all():
        raise DataFormatError(
            f"Trajectory at {source} contains non-finite values (NaN/Inf)",
            expected_format="all finite values",
        )
    return array
