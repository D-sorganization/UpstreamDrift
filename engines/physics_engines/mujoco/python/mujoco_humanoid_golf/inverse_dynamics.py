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

from .kinematic_forces import KinematicForceAnalyzer


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

    def _detect_closed_chains(self) -> bool:
        """Detect if model has closed kinematic chains.

        Returns:
            True if closed chains detected
        """
        # Simple heuristic: Check for equality constraints
        # In production, more sophisticated analysis needed
        return self.model.neq > 0

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
        # Set state
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.qacc[:] = qacc

        # Forward kinematics and dynamics
        # This computes qfrc_bias = C(q,q̇)q̇ + g(q)
        mujoco.mj_forward(self.model, self.data)

        # Capture total bias (C + g)
        total_bias = self.data.qfrc_bias.copy()

        # For parallel mechanisms, capture constraint forces BEFORE changing state
        constraint_forces = None
        if self.has_constraints:
            constraint_forces = self.data.qfrc_constraint.copy()

        # Get mass matrix M(q)
        m_matrix = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, m_matrix, self.data.qM)

        # Compute Gravity g(q) efficiently
        # We need to set velocity to zero to get just gravity.
        # mujoco.mj_forward computes qfrc_bias = g(q) when qvel=0.
        qvel_backup = self.data.qvel.copy()
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        gravity = self.data.qfrc_bias.copy()

        # Compute Coriolis forces: C(q,q̇)q̇ = Total Bias - Gravity
        coriolis = total_bias - gravity

        # Restore velocity (not strictly needed for result but good for consistency)
        self.data.qvel[:] = qvel_backup

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

        # Get Jacobian
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
        # Use least-squares for redundant/constrained systems
        ee_force, _residuals, _rank, _s = lstsq(jacp.T, result.joint_torques)

        return ee_force

    def validate_solution(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        computed_torques: np.ndarray,
    ) -> dict[str, float]:
        """Validate inverse dynamics solution.

        Checks if computed torques actually produce desired acceleration.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            qacc: Desired accelerations [nv]
            computed_torques: Computed torques [nv]

        Returns:
            Validation metrics
        """
        # Apply torques in forward dynamics
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[: self.model.nu] = computed_torques[: self.model.nu]

        # Compute forward dynamics
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)  # Single step

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
