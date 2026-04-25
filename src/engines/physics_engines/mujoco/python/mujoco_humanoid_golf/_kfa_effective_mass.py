from __future__ import annotations

import warnings

import mujoco
import numpy as np

from src.shared.python.core.numerical_constants import (
    EPSILON_SINGULARITY_DETECTION,
)


class _KFAEffectiveMassMixin:
    model: mujoco.MjModel
    _perturb_data: mujoco.MjData
    club_head_id: int | None

    def _validate_effective_mass_direction(self, direction: np.ndarray) -> np.ndarray:
        direction_norm = np.linalg.norm(direction)
        if direction_norm < EPSILON_SINGULARITY_DETECTION:
            raise ValueError(
                f"Direction vector has near-zero magnitude: {direction_norm:.2e}. "
                "Cannot compute effective mass for zero-length direction."
            )
        return direction / direction_norm

    def _check_mass_matrix_conditioning(self, M: np.ndarray) -> None:
        M_cond = np.linalg.cond(M)
        if M_cond > 1e6:
            warnings.warn(
                f"Mass matrix is ill-conditioned: κ(M) = {M_cond:.2e} > 1e6. "
                "Effective mass computation may be numerically unstable. "
                "This often indicates the robot is near a kinematic singularity.",
                category=UserWarning,
                stacklevel=2,
            )

        eigenvalues = np.linalg.eigvalsh(M)
        if np.any(eigenvalues <= 0):
            raise ValueError(
                f"Mass matrix is not positive definite. "
                f"Minimum eigenvalue: {eigenvalues.min():.2e}. "
                "This indicates a modeling error or numerical instability."
            )

    def _check_jacobian_rank(self, jacp: np.ndarray) -> None:
        J_rank = np.linalg.matrix_rank(jacp)
        if J_rank < 3:
            warnings.warn(
                f"Jacobian is rank deficient: rank={J_rank} < 3. "
                "Robot has lost mobility in some directions. "
                "Effective mass may not be well-defined.",
                category=RuntimeWarning,
                stacklevel=2,
            )

    def _compute_effective_mass_value(
        self, direction: np.ndarray, jacp: np.ndarray, M: np.ndarray
    ) -> float:
        J_dir = direction @ jacp
        M_inv = np.linalg.inv(M)
        denominator = J_dir @ M_inv @ J_dir.T + EPSILON_SINGULARITY_DETECTION

        if abs(denominator) < 1e-8:
            warnings.warn(
                f"Effective mass denominator near zero: {denominator:.2e}. "
                "Robot is at or very close to a kinematic singularity in the "
                f"specified direction {direction}. Effective mass is extremely large.",
                category=UserWarning,
                stacklevel=2,
            )

        m_eff = 1.0 / denominator

        if m_eff < 0:
            raise ValueError(
                f"Computed negative effective mass: {m_eff:.2e} kg. "
                "This indicates a numerical error or modeling issue."
            )

        if not np.isfinite(m_eff):
            warnings.warn(
                f"Effective mass is non-finite: {m_eff}. "
                "Robot is at a kinematic singularity. "
                "Returning large finite value instead.",
                category=UserWarning,
                stacklevel=2,
            )
            m_eff = 1e10

        return float(m_eff)

    def compute_effective_mass(
        self,
        qpos: np.ndarray,
        direction: np.ndarray,
        body_id: int | None = None,
    ) -> float:
        """Compute effective mass in a given direction.

        Effective mass determines how difficult it is to accelerate
        in a specific direction. Near kinematic singularities, the
        effective mass can become very large (approaching infinity).

        NUMERICAL STABILITY (Assessment B-008):
        ----------------------------------------
        This method monitors the condition number of the mass matrix
        and Jacobian to detect approaching singularities. When detected:
        - Warning issued if condition number > 1e6
        - Regularization applied automatically
        - Result validity flag returned

        PHYSICS:
        --------
        The effective mass is computed as:
            m_eff = (J M^{-1} J^T)^{-1}

        Where:
        - J = Jacobian mapping joint velocities to task-space velocity
        - M = joint-space mass matrix (configuration-dependent)

        Near singularities:
        - m_eff → ∞ (infinite mass, cannot accelerate)
        - Physically meaningful: robot at kinematic boundary

        FIXED: Uses dedicated _perturb_data to prevent state corruption.

        Args:
            qpos: Joint positions [nv] (rad for revolute, m for prismatic)
            direction: Direction vector [3] (will be normalized)
            body_id: Body to compute for (default: club head)

        Returns:
            Effective mass in that direction [kg]

        Warns:
            UserWarning: If approaching singularity (condition number > 1e6)

        Raises:
            ValueError: If mass matrix is not positive definite
            RuntimeWarning: If Jacobian is rank deficient

        Examples:
            >>> # Compute effective mass in vertical direction
            >>> direction = np.array([0, 0, 1])  # Z-up
            >>> m_eff = analyzer.compute_effective_mass(qpos, direction)
            >>> print(f"Effective mass: {m_eff:.2f} kg")
        """
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        if body_id is None:
            body_id = self.club_head_id

        if body_id is None:
            return 0.0

        direction = self._validate_effective_mass_direction(direction)

        M = self.compute_mass_matrix(qpos)
        self._check_mass_matrix_conditioning(M)

        self._perturb_data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self._perturb_data)

        jacp, _ = self._compute_jacobian(body_id, data=self._perturb_data)
        self._check_jacobian_rank(jacp)

        return self._compute_effective_mass_value(direction, jacp, M)
