"""Python mirror of MATLAB ``+validators/*`` package.

Each function raises :class:`ValueError` (matching the project's DbC style)
with a descriptive message; the MATLAB error identifiers are recorded in
the docstrings so callers can grep across both languages.

Public API:
    must_have_fields         -- ``mustHaveFields.m``
    must_be_finite_vector    -- ``mustBeFiniteVector.m``
    must_be_monotonic_time   -- ``mustBeMonotonicTime.m``
    must_be_unit_quaternion_rows -- ``mustBeUnitQuaternionRows.m``
    must_be_regularizer_kind -- ``mustBeRegularizerKind.m``
    REGULARIZER_KINDS        -- frozen set of allowed regularizer names.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.vector_math import row_euclidean_norm

__all__ = [
    "REGULARIZER_KINDS",
    "must_be_finite_vector",
    "must_be_monotonic_time",
    "must_be_regularizer_kind",
    "must_be_unit_quaternion_rows",
    "must_have_fields",
]

REGULARIZER_KINDS: frozenset[str] = frozenset(
    {
        "total_work",
        "peak_power",
        "torque_l2",
        "coeff_l2",
        "effort_l2",
        "smoothness_l2",
    }
)


def must_have_fields(obj: object, names: Iterable[str]) -> None:
    """Raise ``ValueError`` if ``obj`` lacks any of the named attributes/keys.

    Mirrors ``validators/mustHaveFields.m`` (id ``validator:missingField``).
    Accepts either a mapping (e.g. dict) or an arbitrary object with
    attributes (e.g. dataclass / namedtuple).
    """
    requested = list(names)
    if isinstance(obj, dict):
        present = set(obj.keys())
    else:
        present = {n for n in requested if hasattr(obj, n)}
        if not present and hasattr(obj, "__dict__"):
            present = set(vars(obj).keys())
    missing = [n for n in requested if n not in present]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")


def must_be_finite_vector(v: object) -> NDArray[np.float64]:
    """Validate ``v`` is a non-empty real, finite 1-D numeric array.

    Mirrors ``validators/mustBeFiniteVector.m``. Returns the array as a
    ``float64`` ndarray for convenience so call sites can chain.
    """
    if v is None:
        raise ValueError("value must be a real numeric vector (got None)")
    arr = np.asarray(v)
    if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise ValueError("value must be a real numeric array")
    if arr.size == 0 or arr.ndim != 1:
        raise ValueError("value must be a non-empty 1-D vector")
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("value must contain only finite entries (no NaN/Inf)")
    return arr


def must_be_monotonic_time(t: object) -> NDArray[np.float64]:
    """Validate ``t`` is a strictly increasing finite 1-D numeric vector.

    Mirrors ``validators/mustBeMonotonicTime.m``.
    """
    arr = must_be_finite_vector(t)
    if np.any(np.diff(arr) <= 0):
        raise ValueError("time vector must be strictly increasing")
    return arr


def must_be_unit_quaternion_rows(
    q: object,
    tol: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Validate ``q`` is an ``(N, 4)`` real matrix with unit-norm rows.

    Mirrors ``validators/mustBeUnitQuaternionRows.m``.
    """
    if tol <= 0:
        raise ValueError(f"tol must be > 0; got {tol!r}")
    arr = np.asarray(q)
    if (
        not np.issubdtype(arr.dtype, np.number)
        or np.iscomplexobj(arr)
        or arr.ndim != 2
        or arr.shape[1] != 4
        or arr.size == 0
    ):
        raise ValueError(
            f"quaternion array must be a non-empty (N, 4) real matrix; got shape {arr.shape}"
        )
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("quaternion array contains NaN or Inf")
    norms = row_euclidean_norm(arr)
    worst = float(np.max(np.abs(norms - 1.0)))
    if worst > tol:
        raise ValueError(
            f"quaternion rows must be unit-norm to {tol:g}; worst deviation {worst:.3g}"
        )
    return arr


def must_be_regularizer_kind(name: object) -> str:
    """Validate ``name`` is one of the allowed regularizer kinds.

    Mirrors ``validators/mustBeRegularizerKind.m``.
    Returns the lower-cased string for downstream dispatch.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("regularizer name must be a non-empty string")
    s = name.lower()
    if s not in REGULARIZER_KINDS:
        raise ValueError(
            f"regularizer {s!r} is not one of: {', '.join(sorted(REGULARIZER_KINDS))}"
        )
    return s
