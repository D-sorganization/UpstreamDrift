"""Inverse dynamics computation for golf swing analysis.

This module provides inverse dynamics solvers for computing required joint
torques from desired motion. Includes:

- Full inverse dynamics for open-chain systems
- Partial inverse dynamics for parallel mechanisms (closed-chain)
- Recursive Newton-Euler algorithm
- Composite rigid body algorithm
- Force decomposition analysis
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.linalg import lstsq

from .kinematic_forces import KinematicForceAnalyzer, MjDataContext


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


@dataclass
class ForceDecomposition:
    """Decomposition of forces/torques into components."""

    total: np.ndarray  # Total force/torque
    inertial: np.ndarray  # Due to acceleration (Ma)
    coriolis: np.ndarray  # Due to velocity (C(q,q̇)q̇)
    centrifugal: np.ndarray  # Due to centrifugal effects
    gravity: np.ndarray  # Due to gravity (g(q))
    external: np.ndarray | None = None  # External forces


class InverseDynamicsSolver:
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
        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel
        self._perturb_data.qacc[:] = qacc

        # Forward kinematics and dynamics
        # This computes qfrc_bias = C(q,q̇)q̇ + g(q)
        mujoco.mj_forward(self.model, self._perturb_data)

        # Capture total bias (C + g)
        total_bias = self._perturb_data.qfrc_bias.copy()

        # For parallel mechanisms, capture constraint forces BEFORE changing state
        constraint_forces = None
        if self.has_constraints:
            constraint_forces = self._perturb_data.qfrc_constraint.copy()

        # Get mass matrix M(q)
        m_matrix = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, m_matrix, self._perturb_data.qM)

        # Compute Gravity g(q) efficiently
        # We need to set velocity to zero to get just gravity.
        # mujoco.mj_forward computes qfrc_bias = g(q) when qvel=0.
        # OPTIMIZATION: Use mj_rne with qvel=0 instead of full mj_forward.
        # mj_forward computes everything (kinematics, COM, inertia, etc.) which is slow.
        # mj_rne only computes inverse dynamics. When qvel=0 and qacc=0,
        # it returns gravity.
        qvel_backup = self._perturb_data.qvel.copy()
        self._perturb_data.qvel[:] = 0
        # Explicitly zero out spatial velocity (cvel) to ensure RNE uses correct state
        # This is much faster than running mj_fwdVelocity or mj_forward
        self._perturb_data.cvel[:] = 0

        gravity = np.zeros(self.model.nv)
        # flg_acc=0 means ignore qacc (treat as 0)
        mujoco.mj_rne(self.model, self._perturb_data, 0, gravity)

        # Compute Coriolis forces: C(q,q̇)q̇ = Total Bias - Gravity
        coriolis = total_bias - gravity

        # Restore velocity (not strictly needed for result but good for consistency)
        self._perturb_data.qvel[:] = qvel_backup

        # Inverse dynamics: τ = M q̈ + C q̇ + g - τ_ext
        inertial = m_matrix @ qacc
        total_torques = inertial + coriolis + gravity

        if external_forces is not None:
            total_torques -= external_forces

        return InverseDynamicsResult(
            joint_torques=total_torques,
            constraint_forces=constraint_forces,
            inertial_torques=inertial,
            coriolis_torques=coriolis,
            gravity_torques=gravity,
            is_feasible=True,
            residual_norm=0.0,
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

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            ctrl: Applied control torques [nu] (or nv if full actuation)

        Returns:
            InducedAccelerationResult with component accelerations.
        """
        # Set state
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel

        # 1. Compute Mass Matrix M
        mujoco.mj_fullM(
            self.model, np.zeros((self.model.nv, self.model.nv)), self.data.qM
        )
        # However, we can use mj_solveM to solve M*x = y without explicit inverse,
        # which is faster/steadier.
        # But for separation, we want M^-1 * vector.

        # Let's compute Force Vectors first.
        # We need Coriolis and Gravity as TORQUES.

        # A. Gravity
        # mj_rne with v=0, a=0 returns G.
        qvel_backup = self.data.qvel.copy()
        self.data.qvel[:] = 0
        self.data.cvel[:] = 0
        g_force = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, self.data, 0, g_force)

        # B. Coriolis (C * qdot)
        # mj_forward computes qfrc_bias = C + G.
        self.data.qvel[:] = qvel_backup  # Restore
        mujoco.mj_forward(self.model, self.data)
        bias_force = self.data.qfrc_bias.copy()
        c_force = bias_force - g_force

        # C. Control Torques
        # Map ctrl to joint space if needed.
        # If ctrl is smaller than nv, we assume it matches nu.
        # mujoco stores applied forces (passive + active) in qfrc_applied + actuation?
        # Ideally we take 'ctrl' input and map to generalized force.
        # For simplicity, if len(ctrl) == nu, use data.ctrl.
        # We want the effect of 'ctrl' vector.
        # We can use mj_mulJacT? Or just assign data.ctrl and run mj_fwdActuation?
        # Actually easier: The user passes 'ctrl' vector.
        # If we just want impact of 'ctrl', we set data.ctrl = ctrl, run actuation.
        self.data.ctrl[:] = 0
        if len(ctrl) == self.model.nu:
            self.data.ctrl[:] = ctrl
        # Need to compute qfrc_actuation
        mujoco.mj_fwdActuation(self.model, self.data)
        tau_force = self.data.qfrc_actuation.copy()

        # Scale external forces if any? ignoring for now as they are not passed.

        # Now solve M * a = F for each component.
        # Induced Acc = M^-1 * F_generalized.
        # Note equation direction: M*a + C + G = tau
        # M*a = tau - C - G
        # Acc_G = M^-1 * (-G)
        # Acc_C = M^-1 * (-C)
        # Acc_Tau = M^-1 * (tau)

        # Vectors to solve for:
        f_g = -g_force
        f_c = -c_force
        f_t = tau_force

        # Solve using MuJoCo's Cholesky solver (qLD) which is already computed
        # in mj_forward
        # But we need to use mj_solveM which uses qLD.
        # mj_solveM overwrites the input vector with the solution.

        a_g = f_g.copy()
        mujoco.mj_solveM(self.model, self.data, a_g)

        a_c = f_c.copy()
        mujoco.mj_solveM(self.model, self.data, a_c)

        a_t = f_t.copy()
        mujoco.mj_solveM(self.model, self.data, a_t)

        total = a_g + a_c + a_t

        return InducedAccelerationResult(
            gravity=a_g, velocity=a_c, control=a_t, total=total
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
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.qacc[:] = qacc

        # Use MuJoCo's rne function
        result = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, self.data, 0, result)

        return result


def export_inverse_dynamics_to_csv(
    times: np.ndarray,
    results: list[InverseDynamicsResult],
    filepath: str,
) -> None:
    """Export inverse dynamics results to CSV.

    Args:
        times: Time array [N]
        results: List of InverseDynamicsResult
        filepath: Output CSV path
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        nv = len(results[0].joint_torques)
        header = ["time"]

        for i in range(nv):
            header.extend(
                [f"torque_{i}", f"inertial_{i}", f"coriolis_{i}", f"gravity_{i}"],
            )

        header.append("residual_norm")
        writer.writerow(header)

        # Data rows
        for t, result in zip(times, results, strict=False):
            row = [t]

            for i in range(nv):
                row.append(result.joint_torques[i])
                row.append(
                    (
                        result.inertial_torques[i]
                        if result.inertial_torques is not None
                        else 0.0
                    ),
                )
                row.append(
                    (
                        result.coriolis_torques[i]
                        if result.coriolis_torques is not None
                        else 0.0
                    ),
                )
                row.append(
                    (
                        result.gravity_torques[i]
                        if result.gravity_torques is not None
                        else 0.0
                    ),
                )

            row.append(result.residual_norm)
            writer.writerow(row)


class InverseDynamicsAnalyzer:
    """High-level analyzer combining inverse dynamics and kinematic forces.

    This class provides the complete analysis pipeline for understanding
    swing dynamics from motion capture data.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize analyzer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.id_solver = InverseDynamicsSolver(model, data)
        self.kin_analyzer = KinematicForceAnalyzer(model, data)

    def analyze_captured_motion(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ) -> dict:
        """Complete analysis of captured motion.

        This is the main method for analyzing motion capture data.
        Computes both kinematic forces (Coriolis, centrifugal) and
        inverse dynamics (required torques).

        Args:
            times: Time array [N]
            positions: Joint positions [N x nv]
            velocities: Joint velocities [N x nv]
            accelerations: Joint accelerations [N x nv]

        Returns:
            Dictionary with comprehensive analysis
        """
        # Kinematic force analysis
        kinematic_forces = self.kin_analyzer.analyze_trajectory(
            times,
            positions,
            velocities,
            accelerations,
        )

        # Inverse dynamics
        id_results = self.id_solver.solve_inverse_dynamics_trajectory(
            times,
            positions,
            velocities,
            accelerations,
        )

        # Aggregate statistics
        peak_coriolis_power = 0.0
        max_joint_torque = 0.0

        for kf in kinematic_forces:
            peak_coriolis_power = max(peak_coriolis_power, abs(kf.coriolis_power))

        for id_res in id_results:
            max_joint_torque = max(
                max_joint_torque,
                np.max(np.abs(id_res.joint_torques)),
            )

        return {
            "kinematic_forces": kinematic_forces,
            "inverse_dynamics": id_results,
            "statistics": {
                "peak_coriolis_power": peak_coriolis_power,
                "max_joint_torque": max_joint_torque,
                "duration": times[-1] - times[0],
                "num_frames": len(times),
            },
        }

    def compare_swings(self, swing1_data: dict, swing2_data: dict) -> dict:
        """Compare two swing analyses.

        Args:
            swing1_data: First swing analysis
            swing2_data: Second swing analysis

        Returns:
            Comparison metrics
        """
        stats1 = swing1_data["statistics"]
        stats2 = swing2_data["statistics"]

        return {
            "coriolis_power_diff": stats2["peak_coriolis_power"]
            - stats1["peak_coriolis_power"],
            "torque_diff": stats2["max_joint_torque"] - stats1["max_joint_torque"],
            "duration_diff": stats2["duration"] - stats1["duration"],
        }
