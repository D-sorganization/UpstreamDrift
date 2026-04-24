"""
Impulse, work, and power calculations for pendulum models.

Convention: proximal-on-distal (the force/torque that the proximal segment
exerts on the distal segment).

All functions are pure, stateless, and operate on NumPy arrays.  They are
designed to be called per-timestep or vectorised over a trajectory.

References
----------
Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single-timestep helpers (pure functions)
# ---------------------------------------------------------------------------


def angular_power_at(
    joint_torque: float,
    angular_velocity: float,
) -> float:
    """Instantaneous angular power at a joint (proximal on distal).

    P_angular = tau * omega

    Preconditions:
        joint_torque and angular_velocity are finite scalars.
    Postconditions:
        Returns a finite float.
    """
    assert np.isfinite(joint_torque), f"torque must be finite, got {joint_torque}"
    assert np.isfinite(
        angular_velocity
    ), f"omega must be finite, got {angular_velocity}"
    result = float(joint_torque * angular_velocity)
    assert np.isfinite(result), f"angular power is non-finite: {result}"
    return result


def linear_power_at(
    force: np.ndarray,
    velocity: np.ndarray,
) -> float:
    """Instantaneous linear power at a joint (F Â· v).

    Preconditions:
        force and velocity are shape (2,) finite arrays.
    Postconditions:
        Returns a finite float.
    """
    force = np.asarray(force, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    assert force.shape == (2,), f"force must be shape (2,), got {force.shape}"
    assert velocity.shape == (2,), f"velocity must be shape (2,), got {velocity.shape}"
    assert np.all(np.isfinite(force)), f"force must be finite: {force}"
    assert np.all(np.isfinite(velocity)), f"velocity must be finite: {velocity}"
    result = float(np.dot(force, velocity))
    assert np.isfinite(result), f"linear power is non-finite: {result}"
    return result


# ---------------------------------------------------------------------------
# Trajectory-level computations (vectorised over time)
# ---------------------------------------------------------------------------


def angular_power_series(
    torques: np.ndarray,
    angular_velocities: np.ndarray,
) -> np.ndarray:
    """Angular power time series for one joint.

    Parameters
    ----------
    torques : ndarray, shape (N,)
        Joint torque at each timestep.
    angular_velocities : ndarray, shape (N,)
        Joint angular velocity at each timestep.

    Preconditions:
        Same length, all finite.
    Postconditions:
        Returns shape (N,) finite array.
    """
    torques = np.asarray(torques, dtype=float)
    angular_velocities = np.asarray(angular_velocities, dtype=float)
    assert torques.ndim == 1, f"torques must be 1-D, got {torques.ndim}-D"
    assert (
        torques.shape == angular_velocities.shape
    ), f"Shape mismatch: {torques.shape} vs {angular_velocities.shape}"
    assert np.all(np.isfinite(torques)), "torques must be all finite"
    assert np.all(np.isfinite(angular_velocities)), "velocities must be all finite"

    result: np.ndarray = torques * angular_velocities
    assert np.all(np.isfinite(result)), "angular power series has non-finite values"
    return result


def linear_power_series(
    forces: np.ndarray,
    velocities: np.ndarray,
) -> np.ndarray:
    """Linear power time series for one joint.

    Parameters
    ----------
    forces : ndarray, shape (N, 2)
        Force vector [Fx, Fy] at each timestep.
    velocities : ndarray, shape (N, 2)
        Velocity vector [vx, vy] at each timestep.

    Preconditions:
        Same shape, all finite.
    Postconditions:
        Returns shape (N,) finite array.
    """
    forces = np.asarray(forces, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    assert (
        forces.ndim == 2 and forces.shape[1] == 2
    ), f"forces must be (N,2), got {forces.shape}"
    assert (
        forces.shape == velocities.shape
    ), f"Shape mismatch: {forces.shape} vs {velocities.shape}"
    assert np.all(np.isfinite(forces)), "forces must be finite"
    assert np.all(np.isfinite(velocities)), "velocities must be finite"

    # ⚡ Bolt: np.einsum is ~3x faster than np.sum(..., axis=1) and avoids temporary array allocation
    result: np.ndarray = np.einsum("...i,...i->...", forces, velocities)
    assert np.all(np.isfinite(result)), "linear power series has non-finite values"
    return result


def angular_work_series(
    torques: np.ndarray,
    angular_velocities: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """Cumulative angular work via trapezoidal integration.

    W(t) = integral_0^t tau(s) * omega(s) ds

    Preconditions:
        All arrays shape (N,), all finite, time strictly increasing.
    Postconditions:
        Returns shape (N,) with W[0] = 0.
    """
    power = angular_power_series(torques, angular_velocities)
    time = np.asarray(time, dtype=float)
    assert time.shape == power.shape, f"Shape mismatch: {time.shape} vs {power.shape}"
    assert np.all(np.isfinite(time)), "time must be finite"
    if time.size > 1:
        assert np.all(np.diff(time) > 0), "time must be strictly increasing"

    work = np.zeros_like(power)
    if power.size > 1:
        work[1:] = np.cumsum(0.5 * (power[:-1] + power[1:]) * np.diff(time))

    assert np.all(np.isfinite(work)), "angular work has non-finite values"
    return work


def linear_work_series(
    forces: np.ndarray,
    velocities: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """Cumulative linear work via trapezoidal integration.

    W(t) = integral_0^t F(s) Â· v(s) ds

    Preconditions:
        forces, velocities shape (N,2); time shape (N,); all finite.
    Postconditions:
        Returns shape (N,) with W[0] = 0.
    """
    power = linear_power_series(forces, velocities)
    time = np.asarray(time, dtype=float)
    assert time.shape == power.shape, f"Shape mismatch: {time.shape} vs {power.shape}"

    work = np.zeros_like(power)
    if power.size > 1:
        work[1:] = np.cumsum(0.5 * (power[:-1] + power[1:]) * np.diff(time))

    assert np.all(np.isfinite(work)), "linear work has non-finite values"
    return work


def angular_impulse_series(
    torques: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """Cumulative angular impulse via trapezoidal integration.

    J(t) = integral_0^t tau(s) ds

    Preconditions:
        torques shape (N,), time shape (N,), all finite.
    Postconditions:
        Returns shape (N,) with J[0] = 0.
    """
    torques = np.asarray(torques, dtype=float)
    time = np.asarray(time, dtype=float)
    assert torques.ndim == 1, f"torques must be 1-D, got {torques.ndim}-D"
    assert (
        torques.shape == time.shape
    ), f"Shape mismatch: {torques.shape} vs {time.shape}"
    assert np.all(np.isfinite(torques)), "torques must be finite"
    assert np.all(np.isfinite(time)), "time must be finite"

    impulse = np.zeros_like(torques)
    if torques.size > 1:
        impulse[1:] = np.cumsum(0.5 * (torques[:-1] + torques[1:]) * np.diff(time))

    assert np.all(np.isfinite(impulse)), "angular impulse has non-finite values"
    return impulse


def linear_impulse_series(
    forces: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """Cumulative linear impulse (2-D) via trapezoidal integration.

    J(t) = integral_0^t F(s) ds   (component-wise)

    Preconditions:
        forces shape (N, 2), time shape (N,), all finite.
    Postconditions:
        Returns shape (N, 2) with J[0] = [0, 0].
    """
    forces = np.asarray(forces, dtype=float)
    time = np.asarray(time, dtype=float)
    assert (
        forces.ndim == 2 and forces.shape[1] == 2
    ), f"forces must be (N,2), got {forces.shape}"
    assert (
        forces.shape[0] == time.shape[0]
    ), f"Row mismatch: {forces.shape[0]} vs {time.shape[0]}"

    dt = np.diff(time)
    impulse = np.zeros_like(forces)
    if forces.shape[0] > 1:
        avg = 0.5 * (forces[:-1] + forces[1:])
        impulse[1:] = np.cumsum(avg * dt[:, np.newaxis], axis=0)

    assert np.all(np.isfinite(impulse)), "linear impulse has non-finite values"
    return impulse


# ---------------------------------------------------------------------------
# Convenience: extract all quantities from a simulation result
# ---------------------------------------------------------------------------


def compute_all_dynamics(
    time: np.ndarray,
    joint_torques: np.ndarray,
    angular_velocities: np.ndarray,
    joint_forces: np.ndarray,
    joint_linear_velocities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute all impulse, work, and power quantities for one joint.

    Parameters
    ----------
    time : ndarray, shape (N,)
    joint_torques : ndarray, shape (N,)
        Applied torque at this joint.
    angular_velocities : ndarray, shape (N,)
        Angular velocity of the distal segment at this joint.
    joint_forces : ndarray, shape (N, 2)
        Net force (proximal on distal) at this joint.
    joint_linear_velocities : ndarray, shape (N, 2)
        Linear velocity at this joint location.

    Returns
    -------
    dict with keys:
        angular_power, linear_power,
        angular_work, linear_work,
        angular_impulse, linear_impulse
    """
    assert time is not None, "time must be provided"
    logger.debug("Computing all dynamics quantities for %d timesteps", len(time))

    return {
        "angular_power": angular_power_series(joint_torques, angular_velocities),
        "linear_power": linear_power_series(joint_forces, joint_linear_velocities),
        "angular_work": angular_work_series(joint_torques, angular_velocities, time),
        "linear_work": linear_work_series(joint_forces, joint_linear_velocities, time),
        "angular_impulse": angular_impulse_series(joint_torques, time),
        "linear_impulse": linear_impulse_series(joint_forces, time),
    }
