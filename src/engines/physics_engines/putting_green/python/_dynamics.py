from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.engines.physics_engines.putting_green.python.ball_roll_physics import BallState

if TYPE_CHECKING:
    from src.engines.physics_engines.putting_green.python._sim_core import (
        PuttingGreenSimulator,
    )


def compute_mass_matrix(sim: PuttingGreenSimulator) -> np.ndarray:
    """Compute mass matrix (scalar mass for single ball)."""
    return np.eye(2) * sim.ball_mass


def compute_bias_forces(sim: PuttingGreenSimulator) -> np.ndarray:
    """Compute bias forces (friction + slope)."""
    accel = sim._physics.compute_total_acceleration(sim._ball_state)
    return sim.ball_mass * accel


def compute_gravity_forces(sim: PuttingGreenSimulator) -> np.ndarray:
    """Compute gravitational forces from slope."""
    g_accel = sim._physics.compute_slope_acceleration(sim._ball_state.position)
    return sim.ball_mass * g_accel


def compute_inverse_dynamics(
    sim: PuttingGreenSimulator, qacc: np.ndarray
) -> np.ndarray:
    """Compute forces required for given acceleration."""
    return sim.ball_mass * qacc


def compute_jacobian(
    sim: PuttingGreenSimulator, body_name: str
) -> dict[str, np.ndarray] | None:
    """Compute Jacobian (identity for ball)."""
    if body_name is None:
        raise ValueError("body_name must be provided")
    if body_name == "ball":
        return {
            "linear": np.eye(2),
            "angular": np.zeros((1, 2)),
        }
    return None


def compute_drift_acceleration(sim: PuttingGreenSimulator) -> np.ndarray:
    """Compute passive drift acceleration."""
    return sim._physics.compute_total_acceleration(sim._ball_state)


def compute_control_acceleration(
    sim: PuttingGreenSimulator, tau: np.ndarray
) -> np.ndarray:
    """Compute acceleration from applied force."""
    return tau / sim.ball_mass


def compute_ztcf(
    sim: PuttingGreenSimulator, q: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Zero-torque counterfactual (drift only)."""
    if q is None:
        raise ValueError("q must be provided")
    temp_state = BallState(q, v, sim._ball_state.spin)
    return sim._physics.compute_total_acceleration(temp_state)


def compute_zvcf(sim: PuttingGreenSimulator, q: np.ndarray) -> np.ndarray:
    """Zero-velocity counterfactual."""
    return sim._physics.compute_slope_acceleration(q)
