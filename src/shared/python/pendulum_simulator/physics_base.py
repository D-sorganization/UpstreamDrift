"""
Shared physics utilities for all pendulum models (DRY â€” #C1).

Consolidates duplicated calculations across physics.py, physics_triple.py,
and golfer_*.py into generic N-DOF functions.

Design by Contract
------------------
- All public functions validate inputs with assertions.
- Array shapes are checked to prevent silent broadcasting bugs.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Generic kinetic energy (works for any DOF count)
# ---------------------------------------------------------------------------


def kinetic_energy_from_M(M: np.ndarray, qdot: np.ndarray) -> float:
    """Compute T = 0.5 * qdot^T @ M @ qdot for an arbitrary mass matrix.

    Parameters
    ----------
    M : np.ndarray, shape (n, n)
        Symmetric positive-(semi)definite mass matrix.
    qdot : np.ndarray, shape (n,)
        Generalized velocity vector.

    Returns
    -------
    float
        Kinetic energy (non-negative).

    Pre:
        M is square, M.shape[0] == qdot.shape[0], all finite.
    Post:
        Result is finite and >= 0 (up to floating-point noise).
    """
    n = qdot.shape[0]
    assert M.shape == (n, n), f"M shape {M.shape} vs qdot shape {qdot.shape}"
    assert np.all(np.isfinite(M)), "Mass matrix has non-finite values"
    assert np.all(np.isfinite(qdot)), "Velocity has non-finite values"
    T = float(0.5 * qdot @ M @ qdot)
    assert np.isfinite(T), f"Kinetic energy is non-finite: {T}"
    return T


# ---------------------------------------------------------------------------
# Generic total energy wrapper
# ---------------------------------------------------------------------------


def total_energy_from_parts(kinetic: float, potential: float) -> float:
    """E = T + V.

    Pre: both finite.
    Post: result finite.
    """
    assert np.isfinite(kinetic), f"Kinetic energy non-finite: {kinetic}"
    assert np.isfinite(potential), f"Potential energy non-finite: {potential}"
    return kinetic + potential


# ---------------------------------------------------------------------------
# Generic friction torque (viscous + Coulomb)
# ---------------------------------------------------------------------------


def friction_torque_ndof(
    qdot: np.ndarray,
    viscous_coeffs: np.ndarray,
    coulomb_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """Compute N-DOF dissipative friction torque.

    tau_i = -b_i * qdot_i - mu_i * sign(qdot_i)

    Parameters
    ----------
    qdot : shape (n,) â€” generalized velocities.
    viscous_coeffs : shape (n,) â€” viscous damping per DOF.
    coulomb_coeffs : shape (n,) or None â€” Coulomb friction per DOF.

    Pre: all finite, shapes match.
    Post: opposes motion direction element-wise.
    """
    n = qdot.shape[0]
    assert viscous_coeffs.shape == (
        n,
    ), f"viscous shape {viscous_coeffs.shape} vs qdot {qdot.shape}"
    assert np.all(np.isfinite(qdot)), "qdot has non-finite values"

    tau: npt.NDArray[np.float64] = np.asarray(-viscous_coeffs * qdot, dtype=np.float64)
    if coulomb_coeffs is not None:
        assert coulomb_coeffs.shape == (n,)
        tau -= coulomb_coeffs * np.sign(qdot)

    assert np.all(np.isfinite(tau)), f"Friction torque non-finite: {tau}"
    return tau


# ---------------------------------------------------------------------------
# Generic torque clamping (N-DOF)
# ---------------------------------------------------------------------------


def clamp_torque_ndof(tau: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Clamp each torque element to [-limit_i, +limit_i].

    Parameters
    ----------
    tau : shape (n,) â€” torque vector.
    limits : shape (n,) â€” per-DOF saturation limits (positive).

    Pre: shapes match, limits > 0 or inf.
    Post: |result[i]| <= limits[i].
    """
    n = tau.shape[0]
    assert limits.shape == (n,), f"limits shape {limits.shape} vs tau {tau.shape}"
    assert np.all(limits > 0), "Torque limits must be positive"
    result: npt.NDArray[np.float64] = np.asarray(
        np.clip(tau, -limits, limits), dtype=np.float64
    )
    return result


# ---------------------------------------------------------------------------
# Generic chain forward kinematics
# ---------------------------------------------------------------------------


def chain_positions(
    absolute_angles: np.ndarray,
    lengths: np.ndarray,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Compute 2D joint positions for an open kinematic chain.

    Each segment i has absolute angle absolute_angles[i] measured from
    downward vertical (positive counter-clockwise), and length lengths[i].

    Parameters
    ----------
    absolute_angles : shape (n,)
    lengths : shape (n,)
    origin : (x0, y0) starting point.

    Returns
    -------
    np.ndarray, shape (n, 2) â€” endpoint of each segment.

    Convention: angle 0 = straight down.
        x += -L * sin(angle)
        y += -L * cos(angle)   [y-down convention]
    OR for y-up (this module):
        x += -L * sin(angle)
        y +=  L * cos(angle)

    Note: Caller must handle sign convention.  This function uses the
    convention y-up (positive cosine), matching the existing physics modules.
    """
    n = absolute_angles.shape[0]
    assert lengths.shape == (
        n,
    ), f"lengths {lengths.shape} vs angles {absolute_angles.shape}"

    positions = np.zeros((n, 2))
    x, y = origin
    for i in range(n):
        x += -lengths[i] * np.sin(absolute_angles[i])
        y += -lengths[i] * np.cos(absolute_angles[i])
        positions[i] = (x, y)

    return positions


# ---------------------------------------------------------------------------
# Generic potential energy for pendulum chains
# ---------------------------------------------------------------------------


def potential_energy_chain(
    absolute_angles: np.ndarray,
    lengths: np.ndarray,
    masses: np.ndarray,
    g: float,
) -> float:
    """Compute V for a serial pendulum chain (point masses at segment tips).

    V = -sum_i(cumulative_mass_below_i * g * L_i * cos(angle_i))

    This is the standard pendulum PE where each segment tip carries
    the weight of all segments below it.

    Parameters
    ----------
    absolute_angles : shape (n,) â€” absolute angle of each segment.
    lengths : shape (n,) â€” length of each segment.
    masses : shape (n,) â€” point mass at the end of each segment.
    g : float â€” gravitational acceleration.

    Pre: all finite, shapes match, g >= 0.
    Post: result finite.
    """
    n = absolute_angles.shape[0]
    assert lengths.shape == (n,) and masses.shape == (n,)
    assert g >= 0, f"g must be non-negative, got {g}"

    # Cumulative mass from tip backwards: mass_below[i] = sum(masses[i:])
    # Each segment i contributes: -mass_below_i * g * L_i * cos(angle_i)
    # But for pendulums, the contribution is the total mass that passes
    # through segment i times the height change of that segment.
    V = 0.0
    for i in range(n):
        mass_below = float(np.sum(masses[i:]))
        V -= mass_below * g * lengths[i] * np.cos(absolute_angles[i])

    assert np.isfinite(V), f"Potential energy non-finite: {V}"
    return V


# ---------------------------------------------------------------------------
# Hermite smoothstep (used by joint limit penalties)
# ---------------------------------------------------------------------------


def hermite_smoothstep(x: float) -> float:
    """3rd-order Hermite smoothstep: f(x) = 3xÂ² - 2xÂ³ for x in [0,1].

    Pre: 0 <= x <= 1.
    Post: 0 <= result <= 1, monotonically increasing.
    """
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)
