"""3D-RFT polynomial coefficients (issue #8611).

Source: Agarwal, Goldman and Kamrin, PNAS 120 (2023),
        doi:10.1073/pnas.2214017120.

The stress ratio alpha is computed from three auxiliary functions f1, f2, f3,
each a 20-term polynomial in x1=sin(gamma), x2=cos(beta), x3 (a mixed term).
The coefficient table is transcribed verbatim from Table S1 of the supplement.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DRFT_COEFFICIENTS",
    "compute_alpha_components",
    "compute_f_values",
    "compute_term_basis",
]

# fmt: off
# Coefficient table: 20 rows (terms), 3 columns (c1, c2, c3).
# Each row corresponds to a polynomial term T_k; columns are coefficients for
# f1, f2, f3 respectively. Transcribed from the PNAS supplement Table S1.
#
# PROVENANCE: Agarwal, Goldman and Kamrin, PNAS 120 (2023), Table S1.
# These are BORROWED coefficients fitted to DEM simulations of glass beads in
# a standard frictional-plastic granular medium. They are NOT measurements on
# golf bunker sand.
DRFT_COEFFICIENTS: np.ndarray = np.array([
    # T_k           c1         c2         c3
    # 1             0.00212   -0.06796   -0.02634
    [ 0.00212,  -0.06796,  -0.02634],  # k=1:  1
    [-0.02320,  -0.10941,  -0.03436],  # k=2:  x1
    [-0.20890,   0.04725,   0.45256],  # k=3:  x2
    [-0.43083,  -0.06914,   0.00835],  # k=4:  x3
    [-0.00259,  -0.05835,   0.02553],  # k=5:  x1^2
    [ 0.48872,  -0.65880,  -1.31290],  # k=6:  x2^2
    [-0.00415,  -0.11985,  -0.05532],  # k=7:  x3^2
    [ 0.07204,  -0.25739,   0.06790],  # k=8:  x1*x2
    [-0.02750,  -0.26834,  -0.16404],  # k=9:  x2*x3
    [-0.08772,   0.02692,   0.02287],  # k=10: x3*x1
    [ 0.01992,  -0.00736,   0.02927],  # k=11: x1^3
    [-0.45961,   0.63758,   0.95406],  # k=12: x2^3
    [ 0.40799,   0.08997,  -0.00131],  # k=13: x3^3
    [-0.10107,   0.21069,  -0.11028],  # k=14: x1*x2^2
    [-0.06576,   0.04748,   0.01487],  # k=15: x2*x1^2
    [ 0.05664,   0.20406,  -0.02730],  # k=16: x2*x3^2
    [-0.09269,   0.18519,   0.10911],  # k=17: x3*x2^2
    [ 0.01892,   0.04934,  -0.04097],  # k=18: x3*x1^2
    [ 0.01033,   0.13527,   0.07881],  # k=19: x1*x3^2
    [ 0.15120,  -0.33207,  -0.27519],  # k=20: x1*x2*x3
], dtype=np.float64)
# fmt: on


def compute_term_basis(
    beta: float, gamma: float, psi: float
) -> tuple[float, float, float]:
    """Compute the polynomial basis variables x1, x2, x3.

    Args:
        beta: Surface tilt angle [rad], in [-pi/2, pi/2].
        gamma: Attack angle [rad], in [-pi/2, pi/2].
        psi: Twist angle [rad], in [-pi/2, pi/2].

    Returns:
        (x1, x2, x3) tuple of basis values.
    """
    sin_gamma = np.sin(gamma)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    cos_gamma = np.cos(gamma)
    cos_psi = np.cos(psi)

    x1 = sin_gamma
    x2 = cos_beta
    x3 = cos_psi * cos_gamma * sin_beta + sin_gamma * cos_beta

    return float(x1), float(x2), float(x3)


def _build_term_vector(x1: float, x2: float, x3: float) -> np.ndarray:
    """Build the 20-element vector of polynomial terms T_k."""
    return np.array(
        [
            1.0,  # k=1:  1
            x1,  # k=2:  x1
            x2,  # k=3:  x2
            x3,  # k=4:  x3
            x1 * x1,  # k=5:  x1^2
            x2 * x2,  # k=6:  x2^2
            x3 * x3,  # k=7:  x3^2
            x1 * x2,  # k=8:  x1*x2
            x2 * x3,  # k=9:  x2*x3
            x3 * x1,  # k=10: x3*x1
            x1 * x1 * x1,  # k=11: x1^3
            x2 * x2 * x2,  # k=12: x2^3
            x3 * x3 * x3,  # k=13: x3^3
            x1 * x2 * x2,  # k=14: x1*x2^2
            x2 * x1 * x1,  # k=15: x2*x1^2
            x2 * x3 * x3,  # k=16: x2*x3^2
            x3 * x2 * x2,  # k=17: x3*x2^2
            x3 * x1 * x1,  # k=18: x3*x1^2
            x1 * x3 * x3,  # k=19: x1*x3^2
            x1 * x2 * x3,  # k=20: x1*x2*x3
        ],
        dtype=np.float64,
    )


def compute_f_values(
    beta: float, gamma: float, psi: float
) -> tuple[float, float, float]:
    """Compute the auxiliary functions f1, f2, f3.

    These are 20-term polynomials in the basis variables x1, x2, x3.

    Args:
        beta: Surface tilt angle [rad].
        gamma: Attack angle [rad].
        psi: Twist angle [rad].

    Returns:
        (f1, f2, f3) tuple.
    """
    x1, x2, x3 = compute_term_basis(beta, gamma, psi)
    T = _build_term_vector(x1, x2, x3)

    # f_i = sum_k c_i[k] * T_k
    f1 = float(np.dot(T, DRFT_COEFFICIENTS[:, 0]))
    f2 = float(np.dot(T, DRFT_COEFFICIENTS[:, 1]))
    f3 = float(np.dot(T, DRFT_COEFFICIENTS[:, 2]))

    return f1, f2, f3


def compute_alpha_components(
    beta: float, gamma: float, psi: float
) -> tuple[float, float, float]:
    """Compute the stress ratio components alpha_r, alpha_theta, alpha_z.

    Local cylindrical frame: z_hat up, r_hat = horizontal component of v_hat,
    theta_hat = z_hat x r_hat.

    The stress ratio alpha is dimensionless; it is multiplied by xi_n * |z| to
    get stress [Pa].

    Args:
        beta: Surface tilt angle [rad].
        gamma: Attack angle [rad].
        psi: Twist angle [rad].

    Returns:
        (alpha_r, alpha_theta, alpha_z) tuple.
    """
    f1, f2, f3 = compute_f_values(beta, gamma, psi)

    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    sin_gamma = np.sin(gamma)
    cos_gamma = np.cos(gamma)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    alpha_r = f1 * sin_beta * cos_psi + f2 * cos_gamma
    alpha_theta = f1 * sin_beta * sin_psi
    alpha_z = -f1 * cos_beta - f2 * sin_gamma - f3

    return float(alpha_r), float(alpha_theta), float(alpha_z)
