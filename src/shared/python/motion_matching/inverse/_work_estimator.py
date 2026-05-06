"""Closed-form total-mechanical-work estimate for polynomial torque profiles.

Each joint's torque profile is parameterised by 7 coefficients ``(A..G)``
of a degree-6 polynomial in time:

    tau_j(t) = A*t^6 + B*t^5 + C*t^4 + D*t^3 + E*t^2 + F*t + G

Following APPROACH.md §Loss function we approximate the joint angular
velocity by ``omega_j(t) ≈ d/dt tau_j(t)`` so total work is

    W ≈ Σ_j ∫_0^T |tau_j(t) * d/dt tau_j(t)| dt
       = Σ_j ∫_0^T |0.5 * d/dt (tau_j(t)^2)| dt
       = Σ_j 0.5 * TV_{[0,T]}(tau_j^2)

where ``TV`` is the total variation. Because ``tau_j^2`` is a smooth
polynomial whose extrema occur exactly at the zeros of
``tau_j * tau_j'``, the total variation reduces to the sum of
``|tau_j^2(b) - tau_j^2(a)|`` across the monotonic subintervals between
consecutive extrema.

This module implements that closed-form sum analytically using NumPy's
polynomial root finder; no quadrature is needed. The PyTorch wrapper
``work_estimate_torch`` is differentiable (it internally evaluates the
polynomial at fixed sample points and applies the trapezoidal rule on
``|tau * tau'|``) so the estimate can flow into the training loss.
"""

from __future__ import annotations

import numpy as np
import torch

# Number of coefficients per joint (degree-6 polynomial: A..G).
_COEFFS_PER_JOINT = 7
# Polynomial degree.
_POLY_DEGREE = 6


def analytical_work_per_joint(
    coeffs: np.ndarray,
    *,
    duration_s: float = 1.0,
) -> float:
    """Exact closed-form ``∫_0^T |poly(t) * poly'(t)| dt`` for one joint.

    Parameters
    ----------
    coeffs
        Length-7 array ordered ``[A, B, C, D, E, F, G]`` matching the
        synthetic dataset and ``generateRandomCoefficients.m``.
    duration_s
        Upper integration bound.

    Returns
    -------
    float
        Non-negative scalar work estimate.
    """
    if coeffs.shape != (_COEFFS_PER_JOINT,):
        raise ValueError(
            f"coeffs must have shape ({_COEFFS_PER_JOINT},); got {coeffs.shape}"
        )
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive; got {duration_s}")

    # numpy.polynomial uses ascending powers; our coeffs are descending (A=t^6).
    asc = coeffs[::-1].astype(np.float64)
    poly_sq = np.polynomial.polynomial.polypow(asc, 2)
    deriv = np.polynomial.polynomial.polyder(poly_sq)
    # Extrema of poly^2 are zeros of d/dt(poly^2) = 2 * poly * poly'.
    roots = np.polynomial.polynomial.polyroots(deriv)
    real_roots = roots[np.isreal(roots)].real
    interior = real_roots[(real_roots > 0.0) & (real_roots < duration_s)]
    breakpoints = np.concatenate(
        [np.array([0.0]), np.sort(interior), np.array([duration_s])]
    )
    values = np.polynomial.polynomial.polyval(breakpoints, poly_sq)
    diffs = np.abs(np.diff(values))
    return 0.5 * float(diffs.sum())


def analytical_total_work(
    coefficients: np.ndarray,
    *,
    duration_s: float = 1.0,
) -> float:
    """Sum :func:`analytical_work_per_joint` across all joints.

    Parameters
    ----------
    coefficients
        Either ``(n_joints, 7)`` or flat ``(n_joints * 7,)``.
    duration_s
        Trial duration in seconds.

    Returns
    -------
    float
        Non-negative scalar total work estimate.
    """
    arr = np.asarray(coefficients, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size % _COEFFS_PER_JOINT != 0:
            raise ValueError(
                f"flat coefficients length must be a multiple of "
                f"{_COEFFS_PER_JOINT}; got {arr.size}"
            )
        arr = arr.reshape(-1, _COEFFS_PER_JOINT)
    if arr.ndim != 2 or arr.shape[1] != _COEFFS_PER_JOINT:
        raise ValueError(
            f"coefficients must be 1D or (n_joints, {_COEFFS_PER_JOINT}); "
            f"got shape {arr.shape}"
        )
    return float(
        sum(analytical_work_per_joint(row, duration_s=duration_s) for row in arr)
    )


def work_estimate_torch(
    coefficients: torch.Tensor,
    *,
    duration_s: float = 1.0,
    n_samples: int = 64,
) -> torch.Tensor:
    """Differentiable ``W ≈ ∫|tau * dtau/dt|`` per batch row.

    Used inside the training loss; the trapezoidal rule on a fixed grid is
    a smooth surrogate for the analytical total-variation form and keeps
    autograd happy (no root finding).

    Parameters
    ----------
    coefficients
        ``(B, n_joints * 7)`` flat tensor or ``(B, n_joints, 7)``.
    duration_s
        Upper integration bound.
    n_samples
        Number of trapezoidal sample points along ``[0, duration_s]``.
        Default 64 keeps the estimate within ~1% of the closed form on
        typical degree-6 polynomials.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` non-negative work estimate per batch row.
    """
    if coefficients.dim() not in (2, 3):
        raise ValueError(
            f"coefficients must be 2D or 3D; got {tuple(coefficients.shape)}"
        )
    if coefficients.dim() == 2:
        if coefficients.shape[1] % _COEFFS_PER_JOINT != 0:
            raise ValueError(
                f"flat coefficient dim must be a multiple of {_COEFFS_PER_JOINT}; "
                f"got {coefficients.shape[1]}"
            )
        n_joints = coefficients.shape[1] // _COEFFS_PER_JOINT
        coeff = coefficients.reshape(-1, n_joints, _COEFFS_PER_JOINT)
    else:
        coeff = coefficients

    device = coeff.device
    dtype = coeff.dtype
    t = torch.linspace(0.0, duration_s, n_samples, device=device, dtype=dtype)

    # Powers of t: shape (n_samples, _POLY_DEGREE + 1) descending (t^6..t^0).
    powers_desc = torch.stack(
        [t.pow(_POLY_DEGREE - k) for k in range(_COEFFS_PER_JOINT)],
        dim=1,
    )
    # Derivative powers: d/dt t^k = k * t^(k-1); descending coefficients
    # have multipliers (6, 5, 4, 3, 2, 1, 0).
    deriv_multipliers = torch.tensor(
        [_POLY_DEGREE - k for k in range(_COEFFS_PER_JOINT)],
        device=device,
        dtype=dtype,
    )
    deriv_powers = torch.stack(
        [
            t.pow(max(_POLY_DEGREE - k - 1, 0))
            if (_POLY_DEGREE - k - 1) >= 0
            else torch.zeros_like(t)
            for k in range(_COEFFS_PER_JOINT)
        ],
        dim=1,
    )
    deriv_basis = deriv_powers * deriv_multipliers.unsqueeze(0)

    # tau:    (B, n_joints, n_samples) = einsum coeff * powers_desc
    tau = torch.einsum("bjc,tc->bjt", coeff, powers_desc)
    dtau = torch.einsum("bjc,tc->bjt", coeff, deriv_basis)

    integrand = torch.abs(tau * dtau).sum(dim=1)  # (B, n_samples) — sum joints
    # Trapezoidal rule.
    dt = duration_s / max(n_samples - 1, 1)
    work = 0.5 * dt * (integrand[..., 0] + integrand[..., -1]) + dt * integrand[
        ..., 1:-1
    ].sum(dim=-1)
    return work
