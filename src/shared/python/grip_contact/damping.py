"""Deliberate damping design for the two-bushing grip (issue #11739).

Why a modal design.  The club is one rigid body held by two six-axis bushings
whose points are ``d`` apart along the shaft.  Rotation about an axis across
the grip therefore sees the stiffness ``2 k_r + k_t d^2 / 2`` of the force
pair, and the club mass sits ~0.8 m from the hands, so the lowest modes are
pendulum-like (about 24 Hz for the default stiffness).  A per-axis
``c = 2 zeta sqrt(k m)`` with the club mass leaves those modes nearly
undamped (measured zeta ~ 0.01 for a diagonal-only choice, which is the
ringing seen in the first #11765 run).

Derivation.  Small motion about the club centre of mass with state
``x = (u, theta)``; bushing ``i`` at ``r_i`` (relative to the centre of mass)
with frame ``R_i`` sees the relative displacement ``A_i x`` where
``A_i = [[I, -[r_i]x], [0, I]]``.  Then ``K = sum A_i^T diag(R K_b R^T) A_i``
and likewise ``C`` with the damping coefficients.  The undamped modes
``(K - w^2 M) phi = 0`` give the modal ratio ``zeta_j = phi_j^T C phi_j /
(2 w_j phi_j^T M phi_j)`` which is *linear* in the six per-axis damping
coefficients; the six modal ratios are set to the requested ``zeta`` by
solving that 6x6 non-negative linear system.  No parameter is tuned: the
only inputs are ``zeta`` (default 0.7, the value with ~5 % overshoot and the
fastest settling without sustained ringing) and the geometry, mass and
inertia of the model.  The two bushings share the coefficients.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import nnls

from src.shared.python.grip_contact.club_dynamics import ClubDynamics
from src.shared.python.grip_contact.parameters import BushingParameters

DEFAULT_DAMPING_RATIO = 0.7


def _skew(r: np.ndarray) -> np.ndarray:
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0.0]])


def _blocks(
    params_t: Sequence[float],
    params_r: Sequence[float],
    frames: Sequence[Any],
    com: np.ndarray,
) -> np.ndarray:
    total = np.zeros((6, 6))
    for frame in frames:
        rot = np.asarray(frame.rotation, dtype=float)
        a = np.eye(6)
        a[:3, 3:] = -_skew(np.asarray(frame.position_m, dtype=float) - com)
        local = np.zeros((6, 6))
        local[:3, :3] = rot @ np.diag(params_t) @ rot.T
        local[3:, 3:] = rot @ np.diag(params_r) @ rot.T
        total += a.T @ local @ a
    return total


def assemble_matrices(
    interface: Any, club: ClubDynamics
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(M, K, C)`` of the club on both bushings (6 x 6, about the COM)."""
    com = np.asarray(club.com_m, dtype=float)
    frames = (interface.left, interface.right)
    b = interface.bushing
    mass = np.zeros((6, 6))
    mass[:3, :3] = club.mass_kg * np.eye(3)
    mass[3:, 3:] = club.inertia_com_kg_m2
    k = _blocks(
        b.translational_stiffness_n_m, b.rotational_stiffness_nm_rad, frames, com
    )
    c = _blocks(b.translational_damping_ns_m, b.rotational_damping_nms_rad, frames, com)
    return mass, k, c


def modal_damping(interface: Any, club: ClubDynamics) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frequencies_hz, damping_ratios)`` of the six club modes."""
    mass, k, c = assemble_matrices(interface, club)
    w2, vec = eigh(k, mass)
    omega = np.sqrt(w2)
    ratios = np.array([vec[:, j] @ c @ vec[:, j] / (2.0 * omega[j]) for j in range(6)])
    return omega / (2.0 * math.pi), ratios


def design_damping(
    bushing: BushingParameters,
    left: Any,
    right: Any,
    club: ClubDynamics,
    damping_ratio: float = DEFAULT_DAMPING_RATIO,
) -> BushingParameters:
    """Return ``bushing`` with damping set so every club mode has ``damping_ratio``.

    Raises:
        ValueError: if ``damping_ratio`` is not finite and positive, or if no
            non-negative damping realises it.
    """
    if not math.isfinite(damping_ratio) or damping_ratio <= 0.0:
        raise ValueError(
            f"damping_ratio must be positive and finite, got {damping_ratio}"
        )
    probe = _Probe(left, right, bushing)
    mass, k, _ = assemble_matrices(probe, club)
    w2, vec = eigh(k, mass)
    omega = np.sqrt(w2)
    com = np.asarray(club.com_m, dtype=float)
    gain = np.zeros((6, 6))
    for axis in range(6):
        unit = np.zeros(6)
        unit[axis] = 1.0
        c_axis = _blocks(unit[:3].tolist(), unit[3:].tolist(), (left, right), com)
        for mode in range(6):
            gain[mode, axis] = vec[:, mode] @ c_axis @ vec[:, mode] / (2 * omega[mode])
    coeff, residual = nnls(gain, damping_ratio * np.ones(6))
    if residual > 1e-6 * damping_ratio:
        raise ValueError("no non-negative damping realises the requested ratio")
    return replace(
        bushing,
        translational_damping_ns_m=(float(coeff[0]), float(coeff[1]), float(coeff[2])),
        rotational_damping_nms_rad=(float(coeff[3]), float(coeff[4]), float(coeff[5])),
        source=f"{bushing.source}; modal damping ratio {damping_ratio:g}",
    )


class _Probe:
    """Minimal interface-like holder used before the interface exists."""

    def __init__(self, left: Any, right: Any, bushing: BushingParameters) -> None:
        self.left, self.right, self.bushing = left, right, bushing
