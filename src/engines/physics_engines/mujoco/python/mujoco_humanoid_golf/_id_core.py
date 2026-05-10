from __future__ import annotations

import mujoco
import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

from ._id_models import (
    InducedAccelerationResult,
    InverseDynamicsResult,
)
from ._id_solver_mixin import _InverseDynamicsSolverMixin
from .kinematic_forces import KinematicForceAnalyzer

logger = get_logger(__name__)


class InverseDynamicsSolver(_InverseDynamicsSolverMixin):
    """Solve inverse dynamics for golf swing models.

    This class computes the joint torques required to achieve a desired
    motion trajectory. Handles both open-chain and closed-chain (parallel
    mechanism) systems.

    Key Methods:
    - solve_inverse_dynamics(): Main method for full trajectory
    - compute_required_torques(): Single time step
    - decompose_forces(): Break down torques into components
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize inverse dynamics solver.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        if not (model is not None):
            raise ValueError("model must be provided")
        self.model = model
        self.data = data

        # Initialize kinematic force analyzer
        self.kinematic_analyzer = KinematicForceAnalyzer(model, data)

        # Check if model has constraints (parallel mechanism)
        self.has_constraints = (model.neq > 0) or self._detect_closed_chains()

        # Optimization: Pre-allocate Jacobian arrays and detect API signature
        # This avoids try-except overhead in tight loops
        # (e.g. compute_end_effector_forces)
        self._use_flat_jacobian = False
        try:
            # Test with dummy arrays to check signature
            # Body 0 (world) is always valid
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, 0)
            self._jacp = jacp
            self._jacr = jacr
        except (TypeError, ValueError):
            # Fallback to flat array approach for older MuJoCo bindings
            self._use_flat_jacobian = True
            self._jacp_flat = np.zeros(3 * model.nv)
            self._jacr_flat = np.zeros(3 * model.nv)

        # CRITICAL FIX (Phase 1): Dedicated MjData for thread-safe physics
        # Prevents "Observer Effect" where analysis corrupts visualization state
        self._perturb_data = mujoco.MjData(model)

    def _detect_closed_chains(self) -> bool:
        """Detect if model has closed kinematic chains.

        Returns:
            True if closed chains detected
        """
        # Simple heuristic: Check for equality constraints
        # In production, more sophisticated analysis needed
        return bool(self.model.neq > 0)

    def compute_required_torques(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        external_forces: np.ndarray | None = None,
    ) -> InverseDynamicsResult:
        """Compute required joint torques for desired motion.

        Uses the equation of motion:
        M(q)q̈ + C(q,q̇)q̇ + g(q) = τ + τ_ext

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            qacc: Joint accelerations [nv]
            external_forces: External forces [nv] (optional)

        Returns:
            InverseDynamicsResult with computed torques
        """
        # Set state (Thread-Safe: use private data)
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel
        self._perturb_data.qacc[:] = qacc

        # Forward kinematics and dynamics
        # This computes qfrc_bias = C(q,q̇)q̇ + g(q)
        mujoco.mj_forward(self.model, self._perturb_data)

        # Compute inverse dynamics
        # This computes qfrc_inverse = M(q)q̈ + C(q,q̇)q̇ + g(q) - ext
        mujoco.mj_inverse(self.model, self._perturb_data)

        qfrc_inverse = self._perturb_data.qfrc_inverse.copy()

        # Decompose if needed (optional, for result detail)
        # For now, just return total

        return InverseDynamicsResult(
            joint_torques=qfrc_inverse,
            success=True,
            is_feasible=True,
            # Fill validation metrics if available
        )

    def _compute_gravity_force(self) -> np.ndarray:
        qvel_backup = self._perturb_data.qvel.copy()
        self._perturb_data.qvel[:] = 0
        self._perturb_data.cvel[:] = 0
        g_force = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, self._perturb_data, 0, g_force)
        self._perturb_data.qvel[:] = qvel_backup
        return g_force

    def _compute_coriolis_force(self, g_force: np.ndarray) -> np.ndarray:
        if not (g_force is not None):
            raise ValueError("g_force must be provided")
        mujoco.mj_forward(self.model, self._perturb_data)
        bias_force = self._perturb_data.qfrc_bias.copy()
        return bias_force - g_force

    def _compute_control_force(self, ctrl: np.ndarray) -> np.ndarray:
        if not (ctrl is not None):
            raise ValueError("ctrl must be provided")
        self._perturb_data.ctrl[:] = 0
        if len(ctrl) == self.model.nu:
            self._perturb_data.ctrl[:] = ctrl
        mujoco.mj_fwdActuation(self.model, self._perturb_data)
        return self._perturb_data.qfrc_actuation.copy()

    def _solve_component_accelerations(
        self,
        g_force: np.ndarray,
        c_force: np.ndarray,
        tau_force: np.ndarray,
    ) -> InducedAccelerationResult:
        # Acc_G = M^-1 * (-G), Acc_C = M^-1 * (-C), Acc_Tau = M^-1 * (tau)
        if not (g_force is not None):
            raise ValueError("g_force must be provided")
        a_g = (-g_force).copy()
        mujoco.mj_solveM(self.model, self._perturb_data, a_g)

        a_c = (-c_force).copy()
        mujoco.mj_solveM(self.model, self._perturb_data, a_c)

        a_t = tau_force.copy()
        mujoco.mj_solveM(self.model, self._perturb_data, a_t)

        total = a_g + a_c + a_t
        return InducedAccelerationResult(
            gravity=a_g, velocity=a_c, control=a_t, total=total
        )


class RecursiveNewtonEuler:
    """Recursive Newton-Euler algorithm for inverse dynamics.

    More efficient than matrix-based approach for serial chains.
    Useful for real-time applications.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize RNE solver.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        if not (model is not None):
            raise ValueError("model must be provided")
        self.model = model
        self.data = data

    def compute(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
    ) -> np.ndarray:
        """Compute inverse dynamics using RNE.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            qacc: Joint accelerations [nv]

        Returns:
            Joint torques [nv]
        """
        # MuJoCo's internal RNE is very efficient
        # We use MuJoCo's inverse dynamics
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.qacc[:] = qacc

        # Use MuJoCo's rne function
        result = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, self.data, 0, result)

        return result
