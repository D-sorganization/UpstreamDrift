"""Bound-projection helpers for surrogate inversion.

Per ``option2_nn_surrogate/APPROACH.md`` § "Why bound projection (not penalty)",
the inversion uses a hard ``clamp_`` projection onto the coefficient bounds
after every Adam step. Soft penalties let the surrogate be queried outside
its training distribution, which is exactly the regime where the surrogate is
unreliable (see ``ASSUMPTIONS.md § A1``).
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["clamp_", "default_bounds", "validate_bounds"]


def validate_bounds(
    bounds_low: np.ndarray | torch.Tensor,
    bounds_high: np.ndarray | torch.Tensor,
    coeff_dim: int,
) -> None:
    """Raise ``ValueError`` if bounds are malformed.

    Preconditions:
        * ``bounds_low.shape == bounds_high.shape == (coeff_dim,)``
        * ``bounds_low <= bounds_high`` elementwise.
        * Both arrays are finite.
    """
    low = np.asarray(bounds_low, dtype=np.float64)
    high = np.asarray(bounds_high, dtype=np.float64)
    if low.shape != (coeff_dim,) or high.shape != (coeff_dim,):
        raise ValueError(
            f"bounds must have shape ({coeff_dim},); "
            f"got low={low.shape}, high={high.shape}"
        )
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError("bounds must be finite")
    if np.any(low > high):
        raise ValueError("every bounds_low entry must be <= bounds_high")


def default_bounds(coeff_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return generous symmetric default bounds when none are supplied.

    The surrogate's training data is z-scored coefficients, so a default
    range of [-3, 3] covers ~99.7% of a unit-normal training distribution.
    Callers with real per-dimension MATLAB ``generateRandomCoefficients``
    bounds should pass them explicitly.
    """
    low = -3.0 * np.ones(coeff_dim, dtype=np.float32)
    high = 3.0 * np.ones(coeff_dim, dtype=np.float32)
    return low, high


def clamp_(
    coeffs: torch.Tensor,
    bounds_low: torch.Tensor,
    bounds_high: torch.Tensor,
) -> torch.Tensor:
    """Project ``coeffs`` onto ``[bounds_low, bounds_high]`` *in place*.

    Mirrors ``coeffs.data.clamp_(lo, hi)`` from APPROACH.md but supports
    per-dimension bound vectors via ``torch.maximum``/``torch.minimum``
    (``Tensor.clamp_`` only accepts scalars in older torch versions).
    Returns the same tensor for convenience.
    """
    if coeffs.shape[-1] != bounds_low.shape[0]:
        raise ValueError(
            f"coeffs last dim {coeffs.shape[-1]} != bounds dim {bounds_low.shape[0]}"
        )
    with torch.no_grad():
        torch.maximum(coeffs, bounds_low, out=coeffs)
        torch.minimum(coeffs, bounds_high, out=coeffs)
    return coeffs
