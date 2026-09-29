from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InducedAccelerationResult:
    """Result of induced acceleration analysis."""

    gravity: np.ndarray  # Acceleration induced by gravity
    velocity: np.ndarray  # Acceleration induced by velocity (Coriolis/Centrifugal)
    control: np.ndarray  # Acceleration induced by control torques
    total: np.ndarray  # Total acceleration


@dataclass
class InverseDynamicsResult:
    """Result of inverse dynamics computation."""

    joint_torques: np.ndarray  # [nv] - Required joint torques
    constraint_forces: np.ndarray | None = (
        None  # Constraint forces (if parallel mechanism)
    )

    # Force decomposition
    inertial_torques: np.ndarray | None = None  # Ma term
    coriolis_torques: np.ndarray | None = None  # C(q,q̇)q̇ term
    gravity_torques: np.ndarray | None = None  # g(q) term

    # Task-space forces
    end_effector_force: np.ndarray | None = None  # Force at end-effector

    # Validation metrics
    residual_norm: float = 0.0  # For least-squares solutions
    is_feasible: bool = True  # Whether solution is physically feasible

    # Phase 4 / Advanced Control additions
    success: bool = True
    manipulability_index: float | None = None
    joint_names: list[str] | None = None


@dataclass
class ForceDecomposition:
    """Decomposition of forces/torques into components."""

    total: np.ndarray  # Total force/torque
    inertial: np.ndarray  # Due to acceleration (Ma)
    coriolis: np.ndarray  # Due to velocity (C(q,q̇)q̇)
    centrifugal: np.ndarray  # Due to centrifugal effects
    gravity: np.ndarray  # Due to gravity (g(q))
    external: np.ndarray | None = None  # External forces
