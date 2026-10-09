"""Incremental Coulomb contact with an energetic rebound-work target.

Normal impulse is the integration coordinate. Each increment chooses the
maximum-dissipation tangential impulse within its local Coulomb disk; the
normal compression event and target rebound work are located by bisection.
The energy restitution parameter is not a Newton normal-velocity ratio when
normal/tangential inertia couple. This is still an ideal rigid point contact,
not a calibrated ball-face compliance or groove model.
"""

from __future__ import annotations

import math

import numpy as np

from .incremental_contact_kernel import contact_backend_info, integrate_impulse
from .rigid_contact import (
    RigidContactInput,
    RigidContactResult,
    _cross,
    _finish_contact,
    _prepare_contact,
)


def _tangent_frame(normal: np.ndarray) -> np.ndarray:
    reference = (
        np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    first = _cross(reference, normal)
    first /= np.linalg.norm(first)
    return np.column_stack((first, _cross(normal, first)))


def solve_incremental_contact(
    state: RigidContactInput,
    *,
    energetic_restitution: float,
    steps_per_compression: int = 128,
    use_analytic_fastpath: bool = True,
    backend: str = "auto",
) -> RigidContactResult:
    """Integrate impulse until normal compression and rebound work events.

    `energetic_restitution` is explicit and must equal the value carried in
    the input state; the same numerical assumption then has a distinct law
    interpretation from the instantaneous Newton comparator. Refinement is
    controlled by `steps_per_compression` and must be reported with results.
    """
    if not math.isfinite(energetic_restitution) or not 0 <= energetic_restitution <= 1:
        raise ValueError("energetic restitution must be finite and in [0,1]")
    if abs(energetic_restitution - state.restitution) > 1e-12:
        raise ValueError("energetic restitution must equal input coefficient")
    if (
        isinstance(steps_per_compression, bool)
        or not isinstance(steps_per_compression, int)
        or not 16 <= steps_per_compression <= 4096
    ):
        raise ValueError("steps_per_compression must be an integer in [16,4096]")
    if not isinstance(use_analytic_fastpath, bool):
        raise ValueError("use_analytic_fastpath must be a bool")
    if backend not in ("auto", "python_reference", "numba"):
        raise ValueError("backend must be auto, python_reference, or numba")
    system = _prepare_contact(state)
    n = system.normal
    k = system.effective_inverse_mass
    tangent = _tangent_frame(n)
    ktt = tangent.T @ k @ tangent
    eigenvalues, eigenvectors = np.linalg.eigh(ktt)
    if use_analytic_fastpath:
        normal_tangent_coupling = tangent.T @ k @ n
        normal_mass = float(n @ k @ n)
        compression_impulse = system.approach_mps / normal_mass
        initial_tangent_speed = tangent.T @ system.relative_velocity
        is_block_diagonal = float(
            np.linalg.norm(normal_tangent_coupling)
        ) <= 1e-12 * float(np.linalg.norm(k))
        enough_friction_to_stick = (
            state.friction_coefficient * float(eigenvalues.min()) * compression_impulse
            >= float(np.linalg.norm(initial_tangent_speed)) + 1e-12
        )
        if state.friction_coefficient == 0 or (
            is_block_diagonal and enough_friction_to_stick
        ):
            tangent_impulse = (
                np.zeros(2)
                if state.friction_coefficient == 0
                else np.linalg.solve(ktt, initial_tangent_speed)
            )
            impulse = (
                1 + energetic_restitution
            ) * compression_impulse * n + tangent @ tangent_impulse
            compression_work = 0.5 * system.approach_mps * compression_impulse
            return _finish_contact(
                system,
                impulse,
                friction_regime="analytic_frictionless"
                if state.friction_coefficient == 0
                else "analytic_stick",
                contact_law="incremental_coulomb_energetic",
                compression_work_j=compression_work,
                rebound_work_j=energetic_restitution**2 * compression_work,
                fastpath_used=True,
            )
    impulse, compression_work, rebound_work, slipped = integrate_impulse(
        system.relative_velocity,
        n,
        k,
        tangent,
        ktt,
        eigenvalues,
        eigenvectors,
        state.friction_coefficient,
        energetic_restitution,
        steps_per_compression,
        backend=backend,
    )
    result = _finish_contact(
        system,
        impulse,
        friction_regime="incremental_sliding" if slipped else "incremental_sticking",
        contact_law="incremental_coulomb_energetic",
        compression_work_j=compression_work,
        rebound_work_j=rebound_work,
    )
    if not np.isfinite(result.ball_velocity_mps).all() or float(impulse @ n) <= 0:
        raise RuntimeError("incremental contact produced invalid postimpact state")
    return result
