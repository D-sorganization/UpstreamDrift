from __future__ import annotations

import numpy as np

from src.shared.python.physics._contact_types import (
    ContactPoint,
    PressureVisualizationData,
)


def compute_center_of_pressure(
    contacts: list[ContactPoint],
) -> np.ndarray:
    if not contacts:
        return np.zeros(3)

    total_force = 0.0
    weighted_position = np.zeros(3)

    for c in contacts:
        if c.normal_force > 0:
            total_force += c.normal_force
            weighted_position += c.normal_force * c.position

    if total_force < 1e-10:
        return np.zeros(3)

    return weighted_position / total_force


def compute_grip_torque(
    contacts: list[ContactPoint],
    grip_center: np.ndarray,
) -> np.ndarray:
    if contacts is None:
        raise ValueError("contacts must be provided")
    total_torque = np.zeros(3)

    for c in contacts:
        r = c.position - grip_center
        f = c.normal_force * c.normal + c.tangent_force
        total_torque += np.cross(r, f)

    return total_torque


def compute_pressure_visualization(
    contacts: list[ContactPoint],
    grip_center: np.ndarray,
    grip_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    contact_area: float = 0.01,
) -> PressureVisualizationData:
    if contacts is None:
        raise ValueError("contacts must be provided")
    if not contacts:
        return PressureVisualizationData(
            positions=np.zeros((0, 3)),
            pressures=np.array([]),
            normalized_pressures=np.array([]),
            max_pressure=0.0,
            mean_pressure=0.0,
            grip_axis_positions=np.array([]),
            angular_positions=np.array([]),
        )

    n_contacts = len(contacts)
    area_per_contact = contact_area / n_contacts if n_contacts > 0 else 1.0

    positions = np.array([c.position for c in contacts])
    pressures = np.array(
        [
            c.normal_force / area_per_contact if area_per_contact > 0 else 0.0
            for c in contacts
        ]
    )

    max_pressure = float(np.max(pressures)) if len(pressures) > 0 else 0.0
    mean_pressure = float(np.mean(pressures)) if len(pressures) > 0 else 0.0

    if max_pressure > 0:
        normalized_pressures = pressures / max_pressure
    else:
        normalized_pressures = np.zeros(n_contacts)

    grip_axis = grip_axis / np.linalg.norm(grip_axis)
    relative_pos = positions - grip_center

    grip_axis_positions = np.dot(relative_pos, grip_axis)

    if abs(grip_axis[2]) < 0.9:
        perp1 = np.cross(grip_axis, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(grip_axis, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(grip_axis, perp1)

    x_proj = np.dot(relative_pos, perp1)
    y_proj = np.dot(relative_pos, perp2)
    angular_positions = np.arctan2(y_proj, x_proj)

    return PressureVisualizationData(
        positions=positions,
        pressures=pressures,
        normalized_pressures=normalized_pressures,
        max_pressure=max_pressure,
        mean_pressure=mean_pressure,
        grip_axis_positions=grip_axis_positions,
        angular_positions=angular_positions,
    )
