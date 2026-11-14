"""Fit the Simbody Hollars friction curve to the shared Stribeck law (#13560).

The shared law (Pinocchio reference) is
``mu(r) = [mu_d + (mu_s - mu_d) exp(-r^2)] tanh(r)`` with ``r = v / v_t``.
Simbody's Hollars law is ``mu(r) = min(r, 1) [mu_d + 2 (mu_s' - mu_d) / (1 + r^2)]``.
It is linear in ``mu_s'``, so the least-squares static value over a slip range
has a closed form.  Viscous friction enters both laws identically and drops out.
"""

from __future__ import annotations

import math

import numpy as np

FIT_MAX_RATIO = 10.0
_FIT_SAMPLES = 4000


def shared_law_curve(ratio: np.ndarray, mu_s: float, mu_d: float) -> np.ndarray:
    """Shared regularised Stribeck friction coefficient at ``ratio = v / v_t``."""
    return (mu_d + (mu_s - mu_d) * np.exp(-(ratio**2))) * np.tanh(ratio)


def hollars_curve(ratio: np.ndarray, mu_d: float, mu_s: float) -> np.ndarray:
    """Simbody Hollars friction coefficient at ``ratio = v / v_t``."""
    return np.minimum(ratio, 1.0) * (mu_d + 2.0 * (mu_s - mu_d) / (1.0 + ratio**2))


def matched_static_friction(mu_s: float, mu_d: float) -> float:
    """Hollars static coefficient with least-squares match to the shared law.

    Preconditions: ``mu_s >= 0`` and ``mu_d > 0``, both finite.
    Postcondition: finite; minimises the RMS gap over ``0 < r <= FIT_MAX_RATIO``.
    """
    if not (math.isfinite(mu_s) and math.isfinite(mu_d)) or mu_s < 0 or mu_d <= 0:
        raise ValueError(
            f"friction coefficients must be finite, mu_s >= 0, mu_d > 0; "
            f"got mu_s={mu_s}, mu_d={mu_d}"
        )
    r = np.linspace(1e-6, FIT_MAX_RATIO, _FIT_SAMPLES)
    basis = 2.0 * np.minimum(r, 1.0) / (1.0 + r**2)
    residual = shared_law_curve(r, mu_s, mu_d) - hollars_curve(r, mu_d, mu_d)
    return float(mu_d + (basis @ residual) / (basis @ basis))
