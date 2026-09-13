"""Direct constrained degree-6 polynomial representation on fixed normalized time (#9921).

Provides:
- Exact degree elevation from cubic to degree-6 on a fixed full-swing interval s = t / T_full.
- SVD subspace decomposition of the polynomial evaluation matrix over [0, T_prefix] to
  isolate parameter directions with minimal influence on early motion but maximal influence
  at transition.
- Direct conversion between normalized powers p_0..p_6 and Simscape native A..G coefficients.
"""

from __future__ import annotations

from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]

T_FULL_DEFAULT: float = 1.814
DEGREE: int = 6
COEFFS_PER_ACTUATOR: int = DEGREE + 1  # 7 coefficients: p_0 .. p_6


def _validate_coeffs_shape_and_duration(values: np.ndarray, T_full: float) -> None:
    if values.shape[-1] != COEFFS_PER_ACTUATOR:
        raise ValueError(
            f"Expected shape (..., {COEFFS_PER_ACTUATOR}), got {values.shape}"
        )
    if not np.isfinite(T_full) or T_full <= 0:
        raise ValueError("T_full must be finite and positive")


def simscape_powers_to_normalized(
    simscape_coeffs: Array, *, T_full: float = T_FULL_DEFAULT
) -> Array:
    """Convert Simscape native A..G coefficients [A, B, C, D, E, F, G] to ascending powers of s = t/T_full.

    Simscape evaluates: tau(t) = A*t^6 + B*t^5 + C*t^4 + D*t^3 + E*t^2 + F*t + G.
    Normalized time s = t / T_full => t = s * T_full.
    tau(s) = sum_{j=0}^6 p_j * s^j.
    p_j = simscape_coeffs[6 - j] * (T_full ** j).
    """
    values = np.asarray(simscape_coeffs, dtype=np.float64)
    _validate_coeffs_shape_and_duration(values, T_full)

    powers = np.arange(COEFFS_PER_ACTUATOR, dtype=np.float64)
    scale = T_full**powers
    ascending = values[..., ::-1]
    return ascending * scale


def normalized_to_simscape_powers(
    normalized_coeffs: Array, *, T_full: float = T_FULL_DEFAULT
) -> Array:
    """Convert ascending powers of s = t/T_full [p_0, ..., p_6] to Simscape native A..G.

    A = p_6 / T_full^6, ..., G = p_0.
    Returns array in Simscape order [A, B, C, D, E, F, G].
    """
    values = np.asarray(normalized_coeffs, dtype=np.float64)
    _validate_coeffs_shape_and_duration(values, T_full)

    powers = np.arange(COEFFS_PER_ACTUATOR, dtype=np.float64)
    scale = T_full**powers
    ascending_powers = values / scale
    return ascending_powers[..., ::-1]


def cubic_to_normalized_sextic(
    cubic_native_ag: Array, *, T_full: float = T_FULL_DEFAULT
) -> Array:
    """Elevate a cubic polynomial (given in native 7-coefficient Simscape format where A=B=C=0)
    to degree-6 ascending powers of s = t/T_full.
    """
    return simscape_powers_to_normalized(cubic_native_ag, T_full=T_full)


def build_prefix_svd_basis(
    *,
    T_prefix: float = 0.60,
    T_full: float = T_FULL_DEFAULT,
    n_samples: int = 217,
    degree: int = DEGREE,
) -> tuple[Array, Array, Array]:
    """Compute SVD of the Vandermonde matrix V on [0, T_prefix] for normalized time s = t / T_full.

    V has shape (n_samples, degree + 1).
    V = U @ diag(Sigma) @ W.T
    Returns:
    - W: Orthogonal matrix of shape (degree + 1, degree + 1). Columns w_j are right-singular vectors.
    - Sigma: Singular values array of length degree + 1 in descending order.
    - V: The Vandermonde evaluation matrix.
    """
    times = np.linspace(0.0, T_prefix, n_samples, dtype=np.float64)
    s = times / T_full
    V = np.column_stack([s**j for j in range(degree + 1)])
    U, Sigma, Wt = np.linalg.svd(V, full_matrices=False)
    return Wt.T, Sigma, V


def evaluate_normalized_sextic(
    p: Array, t: Array, *, T_full: float = T_FULL_DEFAULT
) -> Array:
    """Evaluate polynomial with normalized ascending coefficients p at physical times t.

    p: shape (n_channels, 7) or (7,)
    t: shape (n_times,)
    Returns: shape (n_times, n_channels) or (n_times,)
    """
    t_arr = np.asarray(t, dtype=np.float64)
    s = t_arr / T_full
    powers = np.column_stack([s**j for j in range(COEFFS_PER_ACTUATOR)])
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim == 1:
        return np.dot(powers, p_arr)
    return np.dot(powers, p_arr.T)
