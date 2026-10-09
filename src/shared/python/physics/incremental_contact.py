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

from .rigid_contact import (
    RigidContactInput,
    RigidContactResult,
    _finish_contact,
    _prepare_contact,
)


def _tangent_frame(normal: np.ndarray) -> np.ndarray:
    reference = (
        np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    first = np.cross(reference, normal)
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(normal, first)))


def _disk_minimizer(
    target: np.ndarray,
    tangent_mass: np.ndarray,
    *,
    radius: float,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Solve a positive-definite 2D quadratic over a closed impulse disk."""
    if radius <= 0:
        return np.zeros(2), bool(np.linalg.norm(target) > 0)
    free = np.linalg.solve(tangent_mass, target)
    if float(np.linalg.norm(free)) <= radius:
        return free, False
    coefficients = eigenvectors.T @ target
    squared_radius = radius * radius

    def excess(multiplier: float) -> float:
        return float(
            np.sum((coefficients / (eigenvalues + multiplier)) ** 2) - squared_radius
        )

    low, high = 0.0, max(1.0, float(np.linalg.norm(target)) / radius)
    while excess(high) > 0:
        high *= 2
    for _ in range(48):
        middle = (low + high) * 0.5
        if excess(middle) > 0:
            low = middle
        else:
            high = middle
    clipped = eigenvectors @ (coefficients / (eigenvalues + high))
    if float(np.linalg.norm(clipped)) > radius * (1 + 1e-10):
        raise RuntimeError("tangential Coulomb disk postcondition failed")
    return clipped, True


def solve_incremental_contact(
    state: RigidContactInput,
    *,
    energetic_restitution: float,
    steps_per_compression: int = 128,
    use_analytic_fastpath: bool = True,
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
    base_step = system.approach_mps / (float(n @ k @ n) * steps_per_compression)
    if not math.isfinite(base_step) or base_step <= 0:
        raise RuntimeError("incremental normal impulse step is invalid")
    impulse = np.zeros(3)
    compression_work = 0.0
    rebound_work = 0.0
    slipped = False

    def trial(
        current: np.ndarray, step: float
    ) -> tuple[np.ndarray, float, float, bool]:
        relative = system.relative_velocity - k @ current
        normal_before = float(relative @ n)
        tangent_target = tangent.T @ (relative - step * k @ n)
        tangential, is_sliding = _disk_minimizer(
            tangent_target,
            ktt,
            radius=state.friction_coefficient * step,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        delta = step * n + tangent @ tangential
        normal_after = float((relative - k @ delta) @ n)
        return delta, normal_before, normal_after, is_sliding

    for _ in range(10 * steps_per_compression):
        step = base_step
        delta, before, after, is_sliding = trial(impulse, step)
        if after <= 0:
            lo, hi = 0.0, step
            for _ in range(48):
                mid = (lo + hi) * 0.5
                if trial(impulse, mid)[2] > 0:
                    lo = mid
                else:
                    hi = mid
            step = hi
            delta, before, after, is_sliding = trial(impulse, step)
            impulse += delta
            compression_work += 0.5 * (before + after) * step
            slipped |= is_sliding
            break
        impulse += delta
        compression_work += 0.5 * (before + after) * step
        slipped |= is_sliding
    else:
        raise RuntimeError("compression event not reached within impulse horizon")
    if compression_work <= 0:
        raise RuntimeError("normal compression work must be positive")

    target_work = energetic_restitution**2 * compression_work
    for _ in range(10 * steps_per_compression):
        if target_work - rebound_work <= 1e-12:
            break
        step = base_step
        delta, before, after, is_sliding = trial(impulse, step)
        gained = -0.5 * (before + after) * step
        if gained >= target_work - rebound_work:
            lo, hi = 0.0, step
            for _ in range(48):
                mid = (lo + hi) * 0.5
                _, start_mid, end_mid, _ = trial(impulse, mid)
                work_mid = -0.5 * (start_mid + end_mid) * mid
                if work_mid < target_work - rebound_work:
                    lo = mid
                else:
                    hi = mid
            step = hi
            delta, before, after, is_sliding = trial(impulse, step)
            gained = -0.5 * (before + after) * step
        impulse += delta
        rebound_work += gained
        slipped |= is_sliding
    else:
        raise RuntimeError("rebound work target not reached within impulse horizon")
    if abs(rebound_work - target_work) > 1e-8:
        raise RuntimeError("energetic restitution postcondition failed")
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
