"""Coefficient letter-order and time-domain conversion (NM-06 #10621).

Legacy polynomial A..G order must map explicitly to time-domain torques.
Bounded coefficients do not imply bounded torque over a horizon; callers
must evaluate on the native clock.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.dataset_tools.canonical import COEFFICIENT_LETTERS
from src.shared.python.motion_matching.polynomial_torque import (
    evaluate_polynomial_torque,
)

__all__ = [
    "CANONICAL_LETTER_ORDER",
    "coefficients_to_time_domain_torques",
    "require_coefficient_letter_order",
]

CANONICAL_LETTER_ORDER: tuple[str, ...] = COEFFICIENT_LETTERS


def require_coefficient_letter_order(letter_order: Sequence[str]) -> None:
    """Reject any letter order that is not the canonical A..G layout."""
    order = tuple(str(x) for x in letter_order)
    if order != CANONICAL_LETTER_ORDER:
        raise ValueError(
            "coefficient letter_order must be canonical A..G "
            f"{CANONICAL_LETTER_ORDER!r}; got {order!r}"
        )


def coefficients_to_time_domain_torques(
    coeffs: NDArray[np.floating],
    times_s: NDArray[np.floating],
    *,
    letter_order: Sequence[str],
) -> NDArray[np.float64]:
    """Convert ``(n_joints, 7)`` A..G coefficients to ``(T, n_joints)`` torques.

    Design by Contract:
    - ``letter_order`` must equal :data:`CANONICAL_LETTER_ORDER`.
    - ``coeffs`` finite, shape ``(n_joints, 7)``.
    - ``times_s`` finite, 1-D, non-empty.
    """
    require_coefficient_letter_order(letter_order)
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if coeffs_arr.ndim != 2 or coeffs_arr.shape[1] != 7:
        raise ValueError(
            f"coeffs must have shape (n_joints, 7); got {coeffs_arr.shape}"
        )
    if not bool(np.all(np.isfinite(coeffs_arr))):
        raise ValueError("coeffs values must be finite")
    if times.ndim != 1 or times.size < 1:
        raise ValueError("times_s must be a non-empty 1-D array")
    if not bool(np.all(np.isfinite(times))):
        raise ValueError("times_s values must be finite")

    rows: list[NDArray[np.float64]] = []
    for t in times:
        rows.append(evaluate_polynomial_torque(coeffs_arr, float(t)))
    return np.stack(rows, axis=0)
