"""Biomechanical Data Container.

Container for biomechanical measurements at a single time point.
Moved from MuJoCo engine to shared code to support multiple engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BiomechanicalData:
    """Container for biomechanical measurements at a single time point.

    All arrays are indexed by joint/actuator number unless otherwise noted.
    """

    # Time
    time: float = 0.0

    # Joint kinematics
    joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_accelerations: np.ndarray = field(default_factory=lambda: np.array([]))

    # Joint kinetics
    joint_torques: np.ndarray = field(
        default_factory=lambda: np.array([]),
    )  # Applied torques
    joint_forces: np.ndarray = field(
        default_factory=lambda: np.array([]),
    )  # Constraint forces

    # Actuator data
    actuator_forces: np.ndarray = field(default_factory=lambda: np.array([]))
    actuator_powers: np.ndarray = field(default_factory=lambda: np.array([]))

    # Club head data (if available)
    club_head_position: np.ndarray | None = None
    club_head_velocity: np.ndarray | None = None
    club_head_acceleration: np.ndarray | None = None
    club_head_speed: float = 0.0

    # Ground reaction forces (if legs present)
    left_foot_force: np.ndarray | None = None
    right_foot_force: np.ndarray | None = None

    # Total energy
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    total_energy: float = 0.0

    # Center of mass
    com_position: np.ndarray | None = None
    com_velocity: np.ndarray | None = None

    # Angular Momentum (at CoM)
    angular_momentum: np.ndarray | None = None

    # Center of Pressure (global frame, usually on z=0 plane)
    cop_position: np.ndarray | None = None

    # Induced Accelerations (acceleration caused by specific force components)
    # Key: Source Name (e.g. 'gravity', 'velocity_dependent', 'actuator_X'), Value: Acceleration array
    induced_accelerations: dict[str, np.ndarray] = field(default_factory=dict)

    # Task Space Induced Accelerations (e.g. Club Head)
    # Key: Body name (e.g. 'club_head'), Value: Dictionary of components ('gravity', 'velocity', 'control') -> 3D vector
    club_induced_accelerations: dict[str, np.ndarray] | None = None

    # Counterfactual Data (e.g. ZTCF, ZVCF results for this timestep)
    # Key: Counterfactual Name (e.g. 'ztcf_pos', 'zvcf_force'), Value: Data array
    counterfactuals: dict[str, np.ndarray] = field(default_factory=dict)
