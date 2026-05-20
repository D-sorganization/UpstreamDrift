from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ControlMode(Enum):
    """Control mode enumeration."""

    TORQUE = "torque"  # Direct torque control
    IMPEDANCE = "impedance"  # Impedance control
    ADMITTANCE = "admittance"  # Admittance control
    HYBRID = "hybrid"  # Hybrid force-position
    COMPUTED_TORQUE = "computed_torque"  # Computed torque
    TASK_SPACE = "task_space"  # Task-space control


@dataclass
class ImpedanceParameters:
    """Parameters for impedance control."""

    stiffness: np.ndarray  # Stiffness matrix K [n x n] or vector [n]
    damping: np.ndarray  # Damping matrix D [n x n] or vector [n]
    inertia: np.ndarray | None = None  # Inertia matrix M [n x n] (optional)

    def as_matrices(self, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to full matrices.

        Args:
            dim: Dimension of control space

        Returns:
            Tuple of (K_matrix, D_matrix, M_matrix)
        """
        # Stiffness
        if dim is None:
            raise ValueError("dim must be provided")
        k_matrix = (
            np.diag(self.stiffness) if self.stiffness.ndim == 1 else self.stiffness
        )

        # Damping
        d_matrix = np.diag(self.damping) if self.damping.ndim == 1 else self.damping

        # Inertia
        if self.inertia is None:
            m_matrix = np.eye(dim)
        elif self.inertia.ndim == 1:
            m_matrix = np.diag(self.inertia)
        else:
            m_matrix = self.inertia

        return k_matrix, d_matrix, m_matrix


@dataclass
class HybridControlMask:
    """Mask for hybrid force-position control.

    For each DOF: True = force control, False = position control
    """

    force_mask: np.ndarray  # Boolean mask [n]

    def get_position_mask(self) -> np.ndarray:
        """Get complementary position control mask."""
        return ~self.force_mask

    def get_force_selection_matrix(self) -> np.ndarray:
        """Get force selection matrix S_f."""
        return np.diag(self.force_mask.astype(float))

    def get_position_selection_matrix(self) -> np.ndarray:
        """Get position selection matrix S_p."""
        return np.diag(self.get_position_mask().astype(float))
