from dataclasses import dataclass

import numpy as np


@dataclass
class KinematicForceData:
    """Container for kinematic-dependent forces at a single time point."""

    time: float

    # Joint-space forces
    coriolis_forces: np.ndarray  # [nv] - Coriolis and centrifugal forces
    gravity_forces: np.ndarray  # [nv] - Gravitational forces

    # Decomposed components
    centrifugal_forces: np.ndarray | None = None  # [nv] - Pure centrifugal
    velocity_coupling_forces: np.ndarray | None = None  # [nv] - Velocity coupling

    # Task-space forces (end-effector)
    club_head_coriolis_force: np.ndarray | None = None  # [3] - at club head
    club_head_centrifugal_force: np.ndarray | None = None  # [3] - at club head
    club_head_apparent_force: np.ndarray | None = None  # [3] - total apparent force

    # Power contributions
    coriolis_power: float = 0.0  # Power dissipated by Coriolis forces
    centrifugal_power: float = 0.0  # Power from centrifugal effects

    # Kinetic energy contributions
    rotational_kinetic_energy: float = 0.0
    translational_kinetic_energy: float = 0.0
