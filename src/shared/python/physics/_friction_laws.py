from __future__ import annotations

import numpy as np

from src.shared.python.physics._contact_types import (
    SLIP_VELOCITY_THRESHOLD,
    ContactState,
    GripParameters,
)


def check_friction_cone(
    normal_force: float,
    tangent_force: np.ndarray,
    friction_coefficient: float,
) -> bool:
    if normal_force is None:
        raise ValueError("normal_force must be provided")
    tangent_magnitude = np.linalg.norm(tangent_force)
    max_tangent = friction_coefficient * abs(normal_force)
    return bool(tangent_magnitude <= max_tangent)


def compute_slip_direction(
    tangent_force: np.ndarray,
) -> np.ndarray:
    magnitude = np.linalg.norm(tangent_force)
    if magnitude < 1e-10:
        return np.zeros(3)
    return np.asarray(tangent_force / magnitude)


def decompose_contact_force(
    contact_force: np.ndarray,
    contact_normal: np.ndarray,
) -> tuple[float, np.ndarray]:
    if contact_force is None:
        raise ValueError("contact_force must be provided")
    normal_force = float(np.dot(contact_force, contact_normal))
    tangent_force = contact_force - normal_force * contact_normal
    return normal_force, tangent_force


def classify_contact_state(
    normal_force: float,
    tangent_force: np.ndarray,
    slip_velocity: np.ndarray,
    params: GripParameters,
) -> ContactState:
    if normal_force is None:
        raise ValueError("normal_force must be provided")
    if normal_force <= 0:
        return ContactState.NO_CONTACT

    slip_speed = np.linalg.norm(slip_velocity)
    if slip_speed > SLIP_VELOCITY_THRESHOLD:
        return ContactState.SLIPPING

    if check_friction_cone(normal_force, tangent_force, params.static_friction):
        return ContactState.STICKING
    return ContactState.SLIPPING
