from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

DEFAULT_STATIC_FRICTION = 0.8
DEFAULT_DYNAMIC_FRICTION = 0.6
SLIP_VELOCITY_THRESHOLD = 0.001


class ContactState(Enum):
    NO_CONTACT = auto()
    STICKING = auto()
    SLIPPING = auto()


@dataclass
class ContactPoint:
    position: np.ndarray
    normal: np.ndarray
    normal_force: float
    tangent_force: np.ndarray
    slip_velocity: np.ndarray
    state: ContactState
    body_name: str = ""
    contact_id: int = 0


@dataclass
class GripContactState:
    contacts: list[ContactPoint]
    total_normal_force: float
    total_tangent_force: np.ndarray
    num_slipping: int
    num_sticking: int
    center_of_pressure: np.ndarray
    timestamp: float = 0.0


@dataclass
class GripParameters:
    static_friction: float = DEFAULT_STATIC_FRICTION
    dynamic_friction: float = DEFAULT_DYNAMIC_FRICTION
    contact_stiffness: float = 1e5
    contact_damping: float = 1e3
    grip_diameter: float = 0.022
    hand_contact_area: float = 0.01


@dataclass
class GripContactTimestep:
    timestamp: float
    total_normal_force: float
    total_tangent_force_mag: float
    num_contacts: int
    num_slipping: int
    num_sticking: int
    slip_ratio: float
    min_slip_margin: float
    mean_slip_margin: float
    center_of_pressure: np.ndarray
    max_pressure: float
    mean_pressure: float
    contact_forces: np.ndarray
    contact_positions: np.ndarray
    slip_velocities: np.ndarray


@dataclass
class PressureVisualizationData:
    positions: np.ndarray
    pressures: np.ndarray
    normalized_pressures: np.ndarray
    max_pressure: float
    mean_pressure: float
    grip_axis_positions: np.ndarray
    angular_positions: np.ndarray
