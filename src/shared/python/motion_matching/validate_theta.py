"""Cross-engine theta-coefficient validator (CROSS_ENGINE_PARITY_SPEC.md §2.2).

The polynomial-torque coefficient vector ``theta`` is the canonical input
to every engine's ``simulate_with_coefficients`` and a postcondition output
of every ``fit_swing_<engine>``. Per the parity spec it must satisfy three
checks before / after each engine call:

1. **Length**: ``theta.size == n_joints * 7`` (the "(n_joints, 7)" packing
   from §2.2). Any other size is rejected with a descriptive
   expected-vs-got message.
2. **Finiteness**: ``np.all(np.isfinite(theta))`` (no ``NaN`` / ``inf``).
3. **Bounds (optional)**: when ``bounds`` is supplied, each per-letter
   ``(lo, hi)`` pair is enforced. The default empirical / spec bound
   table lives in :data:`THETA_BOUNDS`; engines may pass a tighter or
   looser dict via the ``coefficient_bound_strategy`` toggle from
   PR #4278.

The validator is intentionally framework-agnostic: it returns a contiguous
``float64`` ``ndarray`` so call sites can chain
``theta = validate_theta(...)`` and forward the canonicalised array.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "COEFFS_PER_JOINT",
    "DEFAULT_THETA_BOUND_TABLE",
    "validate_theta",
]

# A polynomial torque has 7 coefficients per joint (A..G for t^0..t^6).
COEFFS_PER_JOINT: int = 7

# Default per-letter bound table, keyed by single-character A..G.
# These mirror :data:`shared.motion_matching.THETA_BOUNDS` (the "spec"
# strategy from PR #4278). Engines requesting "empirical" can scale.
DEFAULT_THETA_BOUND_TABLE: dict[str, tuple[float, float]] = {
    "A": (-1000.0, 1000.0),
    "B": (-1000.0, 1000.0),
    "C": (-500.0, 500.0),
    "D": (-500.0, 500.0),
    "E": (-100.0, 100.0),
    "F": (-100.0, 100.0),
    "G": (-25.0, 25.0),
}


def _coerce_bounds(
    bounds: Mapping[str, tuple[float, float]] | None,
) -> tuple[tuple[str, float, float], ...] | None:
    """Validate the ``bounds`` mapping and return an iterable view."""
    if bounds is None:
        return None
    if not isinstance(bounds, Mapping):
        raise TypeError(
            f"bounds must be a mapping letter->(lo, hi); got {type(bounds).__name__}"
        )
    triples: list[tuple[str, float, float]] = []
    for letter, pair in bounds.items():
        if not isinstance(letter, str) or len(letter) != 1:
            raise ValueError(
                f"bounds keys must be single-character letters; got {letter!r}"
            )
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not all(isinstance(x, (int, float)) for x in pair)
        ):
            raise TypeError(
                f"bounds[{letter!r}] must be a (lo, hi) tuple of floats; got {pair!r}"
            )
        lo, hi = float(pair[0]), float(pair[1])
        if lo > hi:
            raise ValueError(f"bounds[{letter!r}] has lo > hi ({lo} > {hi})")
        triples.append((letter, lo, hi))
    return tuple(triples)


def validate_theta(
    theta: Any,
    *,
    n_joints: int,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    name: str = "theta",
) -> NDArray[np.float64]:
    """Validate ``theta`` per CROSS_ENGINE_PARITY_SPEC.md §2.2.

    Args:
        theta: Array-like (1-D flat ``(n_joints * 7,)`` or 2-D
            ``(n_joints, 7)``) polynomial-torque coefficient vector.
        n_joints: Engine's actuated DOF count. ``theta.size`` must equal
            ``n_joints * 7`` after coercion.
        bounds: Optional per-letter ``{"A": (lo, hi), ...}`` overrides.
            When ``None`` (default) the bounds check is skipped --
            length + finiteness are still enforced. Pass
            :data:`DEFAULT_THETA_BOUND_TABLE` (or a scaled copy from
            ``coefficient_bound_strategy``) to enable bound enforcement.
        name: Variable name for error messages (e.g. ``"theta0"``,
            ``"theta_optimal"``). Default ``"theta"``.

    Returns:
        The validated coefficient vector as a contiguous, flat ``float64``
        ``ndarray`` of shape ``(n_joints * 7,)``. Callers can chain.

    Raises:
        TypeError: ``theta`` cannot be coerced to a numeric array, or
            ``bounds`` has a malformed entry.
        ValueError: length, finiteness, or per-letter bound check fails.
            The message always quotes both the expected and observed
            values so DbC violations are debuggable from the log alone.
    """
    if not isinstance(n_joints, int) or n_joints <= 0:
        raise ValueError(f"n_joints must be a positive int; got {n_joints!r}")
    triples = _coerce_bounds(bounds)

    # --- 1. Coerce to a 1-D float64 array -------------------------------
    try:
        arr = np.ascontiguousarray(theta, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be array-like of floats; got {type(theta).__name__}"
        ) from exc

    if arr.ndim == 2 and arr.shape[1] == COEFFS_PER_JOINT:
        arr = arr.reshape(-1)
    elif arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D (n_joints*7,) or 2-D (n_joints, 7); "
            f"got shape {arr.shape}"
        )

    # --- 2. Length check ------------------------------------------------
    expected = n_joints * COEFFS_PER_JOINT
    if arr.size != expected:
        raise ValueError(
            f"{name} length {arr.size} != n_joints*7 = {n_joints}*{COEFFS_PER_JOINT} "
            f"= {expected}"
        )

    # --- 3. Finiteness check -------------------------------------------
    if not np.all(np.isfinite(arr)):
        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        raise ValueError(
            f"{name} contains non-finite entries (NaN={n_nan}, Inf={n_inf}); "
            "spec §2.2 requires np.all(np.isfinite(theta))"
        )

    # --- 4. Optional bounds check --------------------------------------
    if triples is not None:
        _enforce_bounds(arr, n_joints, triples, name)

    return arr


def _enforce_bounds(
    arr: NDArray[np.float64],
    n_joints: int,
    triples: tuple[tuple[str, float, float], ...],
    name: str,
) -> None:
    """Raise ``ValueError`` if any column of ``arr`` violates ``triples``."""
    m = arr.reshape(n_joints, COEFFS_PER_JOINT)
    tol = 1.0e-9
    for letter, lo, hi in triples:
        col = ord(letter) - ord("A")
        if not 0 <= col < COEFFS_PER_JOINT:
            raise ValueError(
                f"bounds key {letter!r} not in A..G "
                f"(maps to column {col}, valid [0, {COEFFS_PER_JOINT}))"
            )
        column = m[:, col]
        if np.any(column < lo - tol) or np.any(column > hi + tol):
            worst = float(np.max(np.abs(column)))
            raise ValueError(
                f"{name} coefficient {letter!r} (column {col}) violates "
                f"bounds [{lo:g}, {hi:g}]; worst |value| = {worst:g}"
            )
