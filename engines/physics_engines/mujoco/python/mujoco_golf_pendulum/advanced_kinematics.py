"""Advanced kinematics analysis for parallel mechanisms and redundant manipulators.

This module provides state-of-the-art robotics analysis tools including:
- Constraint Jacobian analysis for closed-chain systems
- Manipulability and singularity analysis
- Inverse kinematics solvers
- Task-space control frameworks
- Nullspace projection for redundancy resolution
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.linalg import null_space, pinv, svd


@dataclass
class ManipulabilityMetrics:
    """Metrics for manipulability analysis."""

    manipulability_index: float  # Yoshikawa's manipulability measure
    condition_number: float  # Jacobian condition number
    singular_values: np.ndarray  # Singular values of Jacobian
    is_near_singularity: bool  # True if near singularity
    min_singular_value: float  # Minimum singular value
    max_singular_value: float  # Maximum singular value


@dataclass
class ConstraintJacobianData:
    """Data structure for constraint Jacobian analysis."""

    constraint_jacobian: np.ndarray  # Constraint Jacobian matrix
    nullspace_basis: np.ndarray  # Nullspace basis vectors
    nullspace_dimension: int  # Dimension of nullspace
    rank: int  # Rank of constraint Jacobian
    is_overconstrained: bool  # True if rank > DOF


class AdvancedKinematicsAnalyzer:
    """Advanced kinematics analysis for robotics applications.

    This class provides professional-grade robotics analysis tools for:
    - Parallel mechanisms (golf swing has closed-chain constraints)
    - Manipulability and singularity analysis
    - Inverse kinematics
    - Task-space control
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize advanced kinematics analyzer.

        Args:
            model: MuJoCo model structure
            data: MuJoCo data structure
        """
        self.model = model
        self.data = data

        # Singularity threshold (condition number)
        self.singularity_threshold = 30.0

        # IK solver parameters
        self.ik_damping = 0.01  # Damping factor for DLS
        self.ik_max_iterations = 100
        self.ik_tolerance = 1e-4

        # Find important bodies
        self.club_head_id = self._find_body_id("club_head")
        self.left_hand_id = self._find_body_id("left_hand")
        self.right_hand_id = self._find_body_id("right_hand")

        # Optimization: Determine API version for mj_jacBody to avoid try-except overhead
        # We don't pre-allocate buffers here because compute_body_jacobian returns new arrays
        # Detect if we can use reshaped arrays (MuJoCo 3.x) or need flat arrays
        try:
            test_jacp = np.zeros((3, self.model.nv))
            test_jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, test_jacp, test_jacr, 0)
            self._use_shaped_jac = True
        except TypeError:
            self._use_shaped_jac = False

    def _find_body_id(self, name_pattern: str) -> int | None:
        """Find body ID by name pattern (case-insensitive, partial match)."""
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and name_pattern.lower() in body_name.lower():
                return i
        return None

    def compute_body_jacobian(
        self,
        body_id: int,
        point_offset: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Jacobian for a body (or point on body).

        Args:
            body_id: ID of the body
            point_offset: Offset from body frame origin (default: [0,0,0])

        Returns:
            Tuple of (position_jacobian [3 x nv], rotation_jacobian [3 x nv])
        """
        # MuJoCo 3.3+ may require reshaped arrays
        # Optimized for repeated calls: use np.empty (faster) and avoid try-except
        if self._use_shaped_jac:
            jacp = np.empty((3, self.model.nv))
            jacr = np.empty((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        else:
            jacp_flat = np.empty(3 * self.model.nv)
            jacr_flat = np.empty(3 * self.model.nv)
            mujoco.mj_jacBody(self.model, self.data, jacp_flat, jacr_flat, body_id)
            jacp = jacp_flat.reshape(3, self.model.nv)
            jacr = jacr_flat.reshape(3, self.model.nv)

        # If point_offset is provided, adjust position Jacobian
        # For a point at offset r from a body with angular velocity omega,
        # the velocity is v_point = v_body + omega x r
        # Using r x omega = -omega x r = -skew(r) @ omega, the correct formula is:
        # J_p_offset = J_p_body - skew(offset_world) @ J_r_body
        # Note: point_offset is in body frame, so we must transform to world frame first
        if point_offset is not None:
            # Transform offset from body frame to world frame
            body_rot = self.data.xmat[body_id].reshape(3, 3)
            point_offset_world = body_rot @ point_offset
            # Skew-symmetric matrix of offset (in world coordinates)
            skew_offset = np.array(
                [
                    [0, -point_offset_world[2], point_offset_world[1]],
                    [point_offset_world[2], 0, -point_offset_world[0]],
                    [-point_offset_world[1], point_offset_world[0], 0],
                ],
            )
            # Adjust position Jacobian for offset point (subtract, not add)
            jacp = jacp - skew_offset @ jacr

        return jacp, jacr

    def compute_constraint_jacobian(self) -> ConstraintJacobianData:
        """Compute constraint Jacobian for closed-chain analysis.

        For golf swing: Two hands on club creates a closed kinematic chain.
        This is a parallel mechanism requiring constraint analysis.

        Returns:
            ConstraintJacobianData with constraint analysis
        """
        # Get equality constraint Jacobian from MuJoCo
        # MuJoCo stores constraint Jacobians in data.efc_J

        if self.data.nefc == 0:
            # No constraints - return identity
            return ConstraintJacobianData(
                constraint_jacobian=np.eye(self.model.nv),
                nullspace_basis=np.eye(self.model.nv),
                nullspace_dimension=self.model.nv,
                rank=self.model.nv,
                is_overconstrained=False,
            )

        # Extract constraint Jacobian from MuJoCo's sparse format
        # efc_J is stored in compressed sparse row format
        constraint_jac = self.data.efc_J.copy()

        # Reshape to (nefc x nv)
        if len(constraint_jac.shape) == 1:
            # Sparse representation - need to reconstruct
            # For now, use analytical computation for two-hand grip
            constraint_jac = self._compute_grip_constraint_jacobian()

        # Compute nullspace
        try:
            nullspace = null_space(constraint_jac)
        except np.linalg.LinAlgError:
            nullspace = np.zeros((self.model.nv, 0))

        # Compute rank
        rank = np.linalg.matrix_rank(constraint_jac)

        return ConstraintJacobianData(
            constraint_jacobian=constraint_jac,
            nullspace_basis=nullspace,
            nullspace_dimension=nullspace.shape[1],
            rank=rank,
            is_overconstrained=(rank > self.model.nv),
        )

    def _compute_grip_constraint_jacobian(self) -> np.ndarray:
        """Compute constraint Jacobian for two-handed grip.

        Constraint: Both hands must maintain fixed relationship to club.
        This creates a closed-chain constraint.

        Returns:
            Constraint Jacobian matrix
        """
        if self.left_hand_id is None or self.right_hand_id is None:
            return np.zeros((0, self.model.nv))

        # Get Jacobians for both hands
        jacp_left, _ = self.compute_body_jacobian(self.left_hand_id)
        jacp_right, _ = self.compute_body_jacobian(self.right_hand_id)

        # Constraint: relative position should be constant
        # d/dt(p_right - p_left) = 0
        # J_right * qvel - J_left * qvel = 0
        return jacp_right - jacp_left

    def compute_manipulability(
        self,
        jacobian: np.ndarray,
        metric_type: str = "yoshikawa",
    ) -> ManipulabilityMetrics:
        """Compute manipulability metrics for singularity analysis.

        Args:
            jacobian: Jacobian matrix (m x n)
            metric_type: Type of metric ("yoshikawa" or "condition")

        Returns:
            ManipulabilityMetrics with comprehensive analysis
        """
        # Compute SVD
        _U, s, _Vt = svd(jacobian, full_matrices=False)

        # Compute metrics
        if metric_type == "yoshikawa":
            # Yoshikawa's manipulability measure: w = sqrt(det(J*J^T))
            manipulability = np.sqrt(np.abs(np.prod(s)))
        else:
            # Use minimum singular value
            manipulability = s.min() if len(s) > 0 else 0.0

        # Condition number
        min_singular_value_threshold = 1e-10
        condition_number = (
            s.max() / s.min() if s.min() > min_singular_value_threshold else np.inf
        )

        # Check for singularity
        singularity_value_threshold = 1e-3
        is_near_singularity = (
            condition_number > self.singularity_threshold
            or s.min() < singularity_value_threshold
        )

        return ManipulabilityMetrics(
            manipulability_index=manipulability,
            condition_number=condition_number,
            singular_values=s,
            is_near_singularity=is_near_singularity,
            min_singular_value=s.min() if len(s) > 0 else 0.0,
            max_singular_value=s.max() if len(s) > 0 else 0.0,
        )

    def solve_inverse_kinematics(
        self,
        target_body_id: int,
        target_position: np.ndarray,
        target_orientation: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        nullspace_objective: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, int]:
        """Solve inverse kinematics using Damped Least-Squares (DLS).

        This is a professional-grade IK solver with:
        - Damped least-squares for singularity robustness
        - Nullspace projection for redundancy resolution
        - Joint limit avoidance

        Args:
            target_body_id: Body to position
            target_position: Desired position [3]
            target_orientation: Desired orientation quaternion [4] (optional)
            q_init: Initial joint configuration (default: current state)
            nullspace_objective: Desired nullspace configuration (default: current)

        Returns:
            Tuple of (joint_config, success, iterations)
        """
        # Initialize
        q = self.data.qpos.copy() if q_init is None else q_init.copy()

        if nullspace_objective is None:
            nullspace_objective = q.copy()

        # Target task
        use_orientation = target_orientation is not None
        task_dim = 6 if use_orientation else 3

        # Iterative solver
        for iteration in range(self.ik_max_iterations):
            # Set configuration
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)

            # Get current position and orientation
            current_pos = self.data.xpos[target_body_id].copy()
            current_quat = self.data.xquat[target_body_id].copy()

            # Compute position error
            pos_error = target_position - current_pos

            # Compute orientation error if needed
            if use_orientation and target_orientation is not None:
                # Orientation error in axis-angle form
                ori_error = self._compute_orientation_error(
                    current_quat,
                    target_orientation,
                )
                task_error = np.concatenate([pos_error, ori_error])
            else:
                task_error = pos_error

            # Check convergence
            if np.linalg.norm(task_error) < self.ik_tolerance:
                return q, True, iteration

            # Compute Jacobian
            jacp, jacr = self.compute_body_jacobian(target_body_id)

            J = np.vstack([jacp, jacr]) if use_orientation else jacp

            # Damped least-squares inverse
            # dq = J^T (J J^T + λ^2 I)^{-1} e
            damping_matrix = self.ik_damping**2 * np.eye(task_dim)
            j_damped = J.T @ np.linalg.solve(J @ J.T + damping_matrix, task_error)

            # Nullspace projection for redundancy resolution
            # Add nullspace motion toward desired configuration
            # Use rcond for numerical stability (works in all scipy versions)
            # rtol was only added in scipy 1.8.0, so we use rcond for compatibility
            j_pinv = pinv(J, rcond=self.ik_damping)
            nullspace_proj = np.eye(self.model.nv) - j_pinv @ J
            nullspace_motion = nullspace_proj @ (nullspace_objective - q)

            # Combined motion
            alpha_nullspace = 0.1  # Nullspace gain
            dq = j_damped + alpha_nullspace * nullspace_motion

            # Update with line search
            alpha = 1.0
            q_new = q + alpha * dq

            # Clamp to joint limits
            q_new = self._clamp_to_joint_limits(q_new)

            q = q_new

        # Did not converge
        return q, False, self.ik_max_iterations

    def _compute_orientation_error(
        self,
        current_quat: np.ndarray,
        target_quat: np.ndarray,
    ) -> np.ndarray:
        """Compute orientation error in axis-angle form.

        Args:
            current_quat: Current orientation quaternion [w, x, y, z]
            target_quat: Target orientation quaternion [w, x, y, z]

        Returns:
            Orientation error vector [3]
        """
        # Compute relative quaternion
        # q_error = q_target * q_current^{-1}
        q_current_inv = self._quat_conjugate(current_quat)
        q_error = self._quat_multiply(target_quat, q_current_inv)

        # Convert to axis-angle
        # For small rotations: error ≈ 2 * [x, y, z]
        return 2.0 * q_error[1:4]

    def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
        )

    def _clamp_to_joint_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint configuration to joint limits.

        Args:
            q: Joint configuration

        Returns:
            Clamped joint configuration
        """
        q_clamped = q.copy()

        for i in range(min(len(q), self.model.njnt)):
            self.model.jnt_type[i]

            # Only clamp limited joints
            if self.model.jnt_limited[i]:
                q_min = self.model.jnt_range[i, 0]
                q_max = self.model.jnt_range[i, 1]

                # Get qpos index for this joint
                qpos_addr = self.model.jnt_qposadr[i]

                if qpos_addr < len(q_clamped):
                    q_clamped[qpos_addr] = np.clip(q_clamped[qpos_addr], q_min, q_max)

        return q_clamped

    def compute_manipulability_ellipsoid(
        self,
        body_id: int,
        scaling: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute manipulability ellipsoid for visualization.

        The manipulability ellipsoid shows the directional manipulability
        of the end-effector.

        Args:
            body_id: Body to analyze
            scaling: Scaling factor for ellipsoid size

        Returns:
            Tuple of (center [3], radii [3], axes [3x3])
        """
        # Get Jacobian
        jacp, _ = self.compute_body_jacobian(body_id)

        # Compute SVD
        U, s, _Vt = svd(jacp, full_matrices=False)

        # Ellipsoid parameters
        center = self.data.xpos[body_id].copy()
        radii = scaling * s[:3]  # Principal radii
        axes = U[:, :3]  # Principal axes

        return center, radii, axes

    def analyze_singularities(
        self,
        body_id: int,
        q_samples: np.ndarray | None = None,
        num_samples: int = 100,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Analyze workspace for singularities.

        Args:
            body_id: Body to analyze
            q_samples: Joint configurations to sample (default: random)
            num_samples: Number of samples if q_samples not provided

        Returns:
            Tuple of (singular_configs, condition_numbers)
        """
        singular_configs = []
        condition_numbers = []

        # Generate samples if not provided
        if q_samples is None:
            q_samples = self._generate_random_configs(num_samples)

        # Store original configuration
        q_original = self.data.qpos.copy()

        for q in q_samples:
            # Set configuration
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)

            # Compute Jacobian
            jacp, _ = self.compute_body_jacobian(body_id)

            # Compute manipulability
            metrics = self.compute_manipulability(jacp)
            condition_numbers.append(metrics.condition_number)

            # Check if singular
            if metrics.is_near_singularity:
                singular_configs.append(q.copy())

        # Restore original configuration
        self.data.qpos[:] = q_original
        mujoco.mj_forward(self.model, self.data)

        return singular_configs, condition_numbers

    def _generate_random_configs(self, num_samples: int) -> np.ndarray:
        """Generate random joint configurations within joint limits.

        Args:
            num_samples: Number of configurations to generate

        Returns:
            Array of configurations [num_samples x nv]
        """
        configs = []

        for _ in range(num_samples):
            q = np.zeros(self.model.nq)

            for i in range(self.model.njnt):
                if self.model.jnt_limited[i]:
                    q_min = self.model.jnt_range[i, 0]
                    q_max = self.model.jnt_range[i, 1]
                    qpos_addr = self.model.jnt_qposadr[i]
                    q[qpos_addr] = np.random.uniform(q_min, q_max)

            configs.append(q)

        return np.array(configs)

    def compute_nullspace_projection(self, jacobian: np.ndarray) -> np.ndarray:
        """Compute nullspace projection matrix.

        P_null = I - J^+ J

        This projects vectors into the nullspace of J, useful for
        redundancy resolution.

        Args:
            jacobian: Jacobian matrix [m x n]

        Returns:
            Nullspace projection matrix [n x n]
        """
        # Use rcond for numerical stability (works in all scipy versions)
        j_pinv = pinv(jacobian, rcond=1e-3)
        return np.eye(jacobian.shape[1]) - j_pinv @ jacobian

    def compute_task_space_inertia(self, jacobian: np.ndarray) -> np.ndarray:
        """Compute task-space inertia matrix.

        Λ = (J M^{-1} J^T)^{-1}

        This is important for task-space control.

        Args:
            jacobian: Jacobian matrix [m x n]

        Returns:
            Task-space inertia matrix [m x m]
        """
        # Get mass matrix
        m_matrix = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, m_matrix, self.data.qM)

        # Compute M^{-1}
        m_inv = np.linalg.inv(m_matrix)

        # Compute task-space inertia
        return np.linalg.inv(jacobian @ m_inv @ jacobian.T)
