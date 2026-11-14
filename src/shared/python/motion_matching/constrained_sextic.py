"""Direct constrained degree-6 polynomial representation on fixed normalized time (#9921).

Provides:
- Exact degree elevation from cubic to degree-6 on a fixed full-swing interval s = t / T_full.
- SVD subspace decomposition of the polynomial evaluation matrix over [0, T_prefix] to
  isolate parameter directions with minimal influence on early motion but maximal influence
  at transition.
- Direct conversion between normalized ascending powers p_0..p_6 and descending ``polyval``
  coefficients ``[c_6, c_5, c_4, c_3, c_2, c_1, c_0]`` for ``c_6*t^6 + ... + c_0``
  (reversed relative to canonical ascending ``A + B*t + ... + G*t^6`` in ``polynomial_torque.py``).
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


def descending_polyval_to_normalized(
    polyval_coeffs: Array, *, T_full: float = T_FULL_DEFAULT
) -> Array:
    """Convert descending ``polyval`` coefficients ``[c_6, c_5, c_4, c_3, c_2, c_1, c_0]`` to ascending powers of ``s = t/T_full``.

    Evaluates ``tau(t) = c_6*t^6 + c_5*t^5 + c_4*t^4 + c_3*t^3 + c_2*t^2 + c_1*t + c_0``.
    Normalized time ``s = t / T_full`` gives ``tau(s) = sum_{j=0}^6 p_j * s^j``
    with ``p_j = polyval_coeffs[..., 6 - j] * (T_full ** j)``.
    Canonical ``[A..G]`` in ``polynomial_torque.py`` uses the reversed ascending order ``t^0..t^6``.
    """
    values = np.asarray(polyval_coeffs, dtype=np.float64)
    _validate_coeffs_shape_and_duration(values, T_full)

    powers = np.arange(COEFFS_PER_ACTUATOR, dtype=np.float64)
    scale = T_full**powers
    ascending = values[..., ::-1]
    return ascending * scale


def normalized_to_descending_polyval(
    normalized_coeffs: Array, *, T_full: float = T_FULL_DEFAULT
) -> Array:
    """Convert ascending powers of ``s = t/T_full`` ``[p_0, ..., p_6]`` to descending ``polyval`` order.

    Returns ``[c_6, c_5, c_4, c_3, c_2, c_1, c_0]`` with ``c_6 = p_6 / T_full^6`` down to ``c_0 = p_0``.
    """
    values = np.asarray(normalized_coeffs, dtype=np.float64)
    _validate_coeffs_shape_and_duration(values, T_full)

    powers = np.arange(COEFFS_PER_ACTUATOR, dtype=np.float64)
    scale = T_full**powers
    ascending_powers = values / scale
    return ascending_powers[..., ::-1]


def simscape_powers_to_normalized(
    simscape_coeffs: Array, *, T_full: float = T_FULL_DEFAULT
) -> Array:
    """Backward-compatible alias for :func:`descending_polyval_to_normalized`."""
    return descending_polyval_to_normalized(simscape_coeffs, T_full=T_full)


def normalized_to_simscape_powers(
    normalized_coeffs: Array, *, T_full: float = T_FULL_DEFAULT
) -> Array:
    """Backward-compatible alias for :func:`normalized_to_descending_polyval`."""
    return normalized_to_descending_polyval(normalized_coeffs, T_full=T_full)


def cubic_to_normalized_sextic(
    cubic_native_ag: Array | None = None,
    *,
    cubic_descending_coeffs: Array | None = None,
    T_full: float = T_FULL_DEFAULT,
) -> Array:
    """Elevate a descending ``polyval`` cubic (``c_6 = c_5 = c_4 = 0``) to ascending powers of ``s = t/T_full``.

    Preserves the ``cubic_native_ag`` parameter name for backward compatibility
    while also accepting ``cubic_descending_coeffs`` as a keyword alias.
    """
    if cubic_native_ag is not None and cubic_descending_coeffs is not None:
        raise TypeError(
            "Pass only one of cubic_native_ag or cubic_descending_coeffs, not both"
        )
    coeffs = (
        cubic_descending_coeffs
        if cubic_descending_coeffs is not None
        else cubic_native_ag
    )
    if coeffs is None:
        raise TypeError(
            "cubic_to_normalized_sextic() missing required argument: 'cubic_native_ag'"
        )
    return descending_polyval_to_normalized(coeffs, T_full=T_full)


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
