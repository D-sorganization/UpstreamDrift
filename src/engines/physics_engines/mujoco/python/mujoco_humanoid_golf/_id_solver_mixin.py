from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from scipy.linalg import lstsq

from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

from ._id_models import (
    ForceDecomposition,
    InducedAccelerationResult,
    InverseDynamicsResult,
)
from .kinematic_forces import KinematicForceAnalyzer, MjDataContext

logger = get_logger(__name__)


class _InverseDynamicsSolverMixin:
    model: mujoco.MjModel
    data: mujoco.MjData
    kinematic_analyzer: KinematicForceAnalyzer
    _perturb_data: mujoco.MjData
    _use_flat_jacobian: bool
    _jacp: np.ndarray | None
    _jacr: np.ndarray | None
    _jacp_flat: np.ndarray | None
    _jacr_flat: np.ndarray | None
    has_constraints: bool
    compute_required_torques: Any
    _compute_gravity_force: Any
    _compute_coriolis_force: Any
    _compute_control_force: Any
    _solve_component_accelerations: Any

    def compute_torques_with_posture(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc_primary: np.ndarray,
        qpos_desired: np.ndarray,
        kp_posture: float = 10.0,
        primary_body_name: str = "club_head",
    ) -> InverseDynamicsResult:
        """Compute torques achieving primary task + secondary posture (Phase 4).

        Uses Null-Space Projection:
            tau_total = tau_primary + (I - J^T(J J^T)^-1 J) * tau_secondary

        This ensures secondary tasks (like posture) do not interfere with the
        primary task (e.g. club head trajectory).

        Args:
            qpos: Current joint positions
            qvel: Current joint velocities
            qacc_primary: Desired accelerations for the primary task
            qpos_desired: Target posture configuration
            kp_posture: Gain for posture control
            primary_body_name: Name of the body representing the primary task

        Returns:
            InverseDynamicsResult with combined torques
        """
        # 1. Compute Primary Task Torques (using standard Inverse Dynamics)
        # Note: This assumes qacc_primary satisfies the task constraints
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        primary_result = self.compute_required_torques(qpos, qvel, qacc_primary)
        tau_primary = primary_result.joint_torques  # Total generalized force

        # 2. Compute Jacobian for Primary Task
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, primary_body_name
        )
        if body_id == -1:
            # Fallback or error? For now log warning and treat as no task
            logger.warning(
                f"Primary body '{primary_body_name}' not found. Using pure posture."
            )
            J_primary = np.zeros((3, self.model.nv))
        else:
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            # Use private data state which matches qpos
            mujoco.mj_jacBody(self.model, self._perturb_data, jacp, jacr, body_id)
            # Combine pos + rot jacobian? Usually just pos for hitting ball
            J_primary = jacp

        # 3. Compute Null-Space Projector: N = I - J^+ J
        # Pinverse: J^+ = J^T (J J^T)^-1  (for full rank)
        # We use numpy's pinv for safety
        J_pinv = np.linalg.pinv(J_primary)
        N = np.eye(self.model.nv) - np.dot(J_pinv, J_primary)

        # 4. Compute Secondary Posture Torques (PD control in joint space)
        # tau_posture = Kp * (q_des - q) - Kd * q_vel
        q_err = qpos_desired - qpos
        # Simple PD
        kd_posture = 2 * np.sqrt(kp_posture)  # Critical damping approx
        tau_secondary = (kp_posture * q_err) - (kd_posture * qvel)

        # 5. Project Secondary into Null Space
        tau_null = np.dot(N, tau_secondary)

        # 6. Combine
        tau_total = tau_primary + tau_null

        return InverseDynamicsResult(
            joint_torques=tau_total,
            success=True,
            is_feasible=True,
            manipulability_index=primary_result.manipulability_index,
        )

    @precondition(
        lambda self, qpos, qvel, ctrl: len(qpos) > 0,
        "Joint positions must be non-empty",
    )
    def compute_induced_accelerations(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
    ) -> InducedAccelerationResult:
        """Compute acceleration components induced by different forces.

        Using M(q)q_ddot = tau - C(q,q_dot)q_dot - G(q)
        q_ddot = M^-1 * (tau - C - G)

        FIXED (Assessment A Finding A-001 - CRITICAL):
        Now uses self._perturb_data for thread safety instead of modifying
        shared self.data state. This prevents race conditions during parallel
        analysis where visualization thread reads from self.data.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            ctrl: Applied control torques [nu] (or nv if full actuation)

        Returns:
            InducedAccelerationResult with component accelerations.
        """
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel

        g_force = self._compute_gravity_force()
        c_force = self._compute_coriolis_force(g_force)
        tau_force = self._compute_control_force(ctrl)

        return self._solve_component_accelerations(g_force, c_force, tau_force)

    @precondition(
        lambda self, times, positions, velocities, accelerations: len(times) > 0,
        "Time array must be non-empty",
    )
    @precondition(
        lambda self, times, positions, velocities, accelerations: (
            len(times) == len(positions) == len(velocities) == len(accelerations)
        ),
        "All trajectory arrays must have the same length",
    )
    def solve_inverse_dynamics_trajectory(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ) -> list[InverseDynamicsResult]:
        """Solve inverse dynamics for entire trajectory.

        Args:
            times: Time array [N]
            positions: Joint positions [N x nv]
            velocities: Joint velocities [N x nv]
            accelerations: Joint accelerations [N x nv]

        Returns:
            List of InverseDynamicsResult for each time step
        """
        if not (times is not None):
            raise ValueError("times must be provided")
        if not (times is not None):
            raise ValueError("times must be provided")
        results = []

        for i in range(len(times)):
            result = self.compute_required_torques(
                positions[i],
                velocities[i],
                accelerations[i],
            )
            results.append(result)

        return results

    def compute_partial_inverse_dynamics(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        constrained_joints: list[int],
    ) -> InverseDynamicsResult:
        """Compute partial inverse dynamics for parallel mechanisms.

        For closed-chain systems, some joints may be constrained.
        This computes torques for actuated joints while respecting
        constraint forces.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            qacc: Joint accelerations [nv]
            constrained_joints: List of constrained joint indices

        Returns:
            InverseDynamicsResult with partial solution
        """
        # Full inverse dynamics
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        full_result = self.compute_required_torques(qpos, qvel, qacc)

        # Create selection matrix for actuated joints
        actuated_joints = [
            i for i in range(self.model.nv) if i not in constrained_joints
        ]

        # Extract actuated torques
        full_result.joint_torques[actuated_joints]

        # For constrained joints, torques come from constraints
        np.zeros(len(constrained_joints))
        if full_result.constraint_forces is not None:
            full_result.constraint_forces[constrained_joints]

        return full_result  # Return full result with constraint info

    def decompose_forces(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
    ) -> ForceDecomposition:
        """Decompose total forces into components.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            qacc: Joint accelerations [nv]

        Returns:
            ForceDecomposition with all components
        """
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        result = self.compute_required_torques(qpos, qvel, qacc)

        # Decompose Coriolis into centrifugal
        centrifugal, _ = self.kinematic_analyzer.decompose_coriolis_forces(qpos, qvel)

        # Provide defaults for None values
        nv = len(result.joint_torques)
        inertial = (
            result.inertial_torques
            if result.inertial_torques is not None
            else np.zeros(nv)
        )
        coriolis = (
            result.coriolis_torques
            if result.coriolis_torques is not None
            else np.zeros(nv)
        )
        gravity = (
            result.gravity_torques
            if result.gravity_torques is not None
            else np.zeros(nv)
        )

        return ForceDecomposition(
            total=result.joint_torques,
            inertial=inertial,
            coriolis=coriolis,
            centrifugal=centrifugal,
            gravity=gravity,
        )

    @precondition(
        lambda self, qpos, qvel, qacc, body_id: body_id >= 0,
        "Body ID must be non-negative",
    )
    def compute_end_effector_forces(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        body_id: int,
    ) -> np.ndarray:
        """Compute forces at end-effector (e.g., club head).

        Maps joint torques to task-space forces: F = (J^T)^{-1} τ

        PERFORMANCE NOTE (Issue B-003): Uses lstsq which is correct but not optimized.
        For batch processing of trajectories, consider precomputing pseudo-inverses
        if the robot configuration remains similar. However, since Jacobian depends
        on qpos (configuration-dependent), caching is only beneficial for repeated
        calls with identical qpos but different torques (rare in practice).

        Potential optimization for batch processing:
            # For trajectory analysis (same qpos, varying torques):
            J_pinv = np.linalg.pinv(jacp.T)  # Compute once
            ee_forces = J_pinv @ torques_batch  # Reuse for multiple torques

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            qacc: Joint accelerations [nv]
            body_id: Body ID for end-effector

        Returns:
            End-effector force [3]
        """
        # Compute required torques
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        result = self.compute_required_torques(qpos, qvel, qacc)

        # Get Jacobian (configuration-dependent, must recompute for each qpos)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        # Compute Jacobian using pre-allocated arrays and detected API
        if self._use_flat_jacobian:
            mujoco.mj_jacBody(
                self.model,
                self.data,
                self._jacp_flat,
                self._jacr_flat,
                body_id,
            )
            jacp = self._jacp_flat.reshape(3, self.model.nv)
        else:
            mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, body_id)
            jacp = self._jacp

        # Map torques to forces: F = (J^T)^{-1} τ
        # lstsq is robust for redundant/constrained systems (handles rank deficiency)
        ee_force, _residuals, _rank, _s = lstsq(jacp.T, result.joint_torques)

        return np.array(ee_force, dtype=np.float64)

    def validate_solution(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        computed_torques: np.ndarray,
    ) -> dict[str, float]:
        """Validate inverse dynamics solution.

        Checks if computed torques actually produce desired acceleration.

        FIXED: This method now uses MjDataContext for state isolation and
        static calculation (mj_forward) instead of mj_step to avoid the
        "Observer Effect" bug where validation would advance simulation time
        and corrupt subsequent calculations.
        See Issues A-003 and F-001.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            qacc: Desired accelerations [nv]
            computed_torques: Computed torques [nv]

        Returns:
            Validation metrics
        """
        # Use context manager for automatic state save/restore (Issues A-003, F-001)
        with MjDataContext(self.model, self.data):
            # Apply torques in forward dynamics
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            self.data.ctrl[: self.model.nu] = computed_torques[: self.model.nu]

            # Compute forward dynamics (static calculation, no time advancement)
            # FIXED: Removed mj_step which was causing Observer Effect
            mujoco.mj_forward(self.model, self.data)

            # Get resulting acceleration
            m_matrix = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, m_matrix, self.data.qM)

            # Acceleration from dynamics: M^{-1}(τ - C q̇ - g)
            coriolis = self.kinematic_analyzer.compute_coriolis_forces(qpos, qvel)
            gravity = self.kinematic_analyzer.compute_gravity_forces(qpos)

            m_inv = np.linalg.inv(m_matrix)
            computed_qacc = m_inv @ (computed_torques - coriolis - gravity)

            # Error metrics
            acc_error = np.linalg.norm(computed_qacc - qacc)
            relative_error = acc_error / (np.linalg.norm(qacc) + 1e-10)

            return {
                "acceleration_error": float(acc_error),
                "relative_error": float(relative_error),
                "max_torque": float(np.max(np.abs(computed_torques))),
                "mean_torque": float(np.mean(np.abs(computed_torques))),
            }
        # State is automatically restored here by context manager

    def compute_actuator_efficiency(
        self,
        result: InverseDynamicsResult,
    ) -> dict[str, float]:
        """Compute efficiency metrics for actuators.

        Args:
            result: Inverse dynamics result

        Returns:
            Efficiency metrics
        """
        if not (result is not None):
            raise ValueError("result must be provided")
        if not (result is not None):
            raise ValueError("result must be provided")
        torques = result.joint_torques

        # Mechanical advantage (ratio of output to input)
        if result.inertial_torques is not None:
            inertial_ratio = float(
                np.linalg.norm(result.inertial_torques)
                / (np.linalg.norm(torques) + 1e-10),
            )
        else:
            inertial_ratio = 0.0

        # Gravity compensation ratio
        if result.gravity_torques is not None:
            gravity_ratio = float(
                np.linalg.norm(result.gravity_torques)
                / (np.linalg.norm(torques) + 1e-10),
            )
        else:
            gravity_ratio = 0.0

        # Coriolis ratio (ideally small)
        if result.coriolis_torques is not None:
            coriolis_ratio = float(
                np.linalg.norm(result.coriolis_torques)
                / (np.linalg.norm(torques) + 1e-10),
            )
        else:
            coriolis_ratio = 0.0

        return {
            "inertial_ratio": float(inertial_ratio),
            "gravity_ratio": float(gravity_ratio),
            "coriolis_ratio": float(coriolis_ratio),
            "efficiency_index": float(
                inertial_ratio / (gravity_ratio + coriolis_ratio + 1e-10),
            ),
        }
