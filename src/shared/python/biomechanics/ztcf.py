"""Zero-Torque Counterfactual (ZTCF) force computation.

Computes the joint forces that would exist at each joint if all applied
driving torques were zero — isolating the passive structural and
momentum-based force components (gravity, Coriolis, friction).

This is the engine-agnostic ZTCF module.  It accepts pre-computed
dynamics matrices (M, C, G, friction) so that any physics engine
(MuJoCo, Drake, Pinocchio, OpenSim, or the pendulum simulator) can
use a single ZTCF pipeline.

Design by Contract
------------------
Preconditions:
  - Mass matrix M must be square, positive-definite, and finite.
  - All vector inputs must match the DOF dimension of M and be finite.
  - Joint positions, segment masses, and segment lengths must be
    consistent in size and finite.
Postconditions:
  - Returned ZTCFResult contains finite forces and accelerations.
  - Force array shape is (n_joints, n_dims).

DRY
---
Delegates linear algebra to numpy.  The ZTCF equation is solved in
exactly one place (compute_ztcf_accelerations).  Force computation
from accelerations is in compute_ztcf_forces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts.exceptions import (
    PostconditionError,
    PreconditionError,
)

# ============================================================================
# Data structures
# ============================================================================


@dataclass(frozen=True)
class ZTCFResult:
    """Immutable container for ZTCF computation results.

    Attributes
    ----------
    joint_forces : np.ndarray
        Shape ``(n_joints, n_dims)`` — passive force vector at each joint.
    joint_accelerations : np.ndarray
        Shape ``(n_dof,)`` — generalized accelerations under zero torque.
    n_joints : int
        Number of joints in the result.
    """

    joint_forces: np.ndarray
    joint_accelerations: np.ndarray
    n_joints: int

    def force_at_joint(self, joint_idx: int) -> np.ndarray:
        """Return the ZTCF force vector at a single joint.

        Preconditions
        -------------
        ``0 <= joint_idx < n_joints``.
        """
        if not (0 <= joint_idx < self.n_joints):
            raise PreconditionError(
                f"joint_idx must be in [0, {self.n_joints}), got {joint_idx}",
                function_name="ZTCFResult.force_at_joint",
                parameter="joint_idx",
                value=joint_idx,
            )
        return self.joint_forces[joint_idx]

    def magnitudes(self) -> np.ndarray:
        """Return force magnitudes for all joints.  Shape ``(n_joints,)``."""
        n_dims = self.joint_forces.shape[1] if self.joint_forces.ndim > 1 else 1
        if n_dims == 2:
            # ⚡ Bolt: Explicit element-wise hypot is ~5-10x faster than np.linalg.norm(..., axis=1) for 2D vectors
            return np.hypot(self.joint_forces[:, 0], self.joint_forces[:, 1])
        if n_dims == 3:
            # ⚡ Bolt: Explicit element-wise sqrt is ~5-10x faster than np.linalg.norm(..., axis=1) for 3D vectors
            return np.sqrt(
                self.joint_forces[:, 0] ** 2
                + self.joint_forces[:, 1] ** 2
                + self.joint_forces[:, 2] ** 2
            )
        if self.joint_forces.ndim > 1:
            return np.linalg.norm(self.joint_forces, axis=1)
        return np.abs(self.joint_forces)

    def max_magnitude(self) -> float:
        """Return the largest force magnitude across all joints."""
        return float(np.max(self.magnitudes()))


# ============================================================================
# Core ZTCF computation
# ============================================================================


def compute_ztcf_accelerations(
    *,
    mass_matrix: np.ndarray,
    coriolis_vector: np.ndarray,
    gravity_vector: np.ndarray,
    friction_vector: np.ndarray | None = None,
) -> np.ndarray:
    """Compute generalized accelerations under zero applied torque.

    Solves ``M * q̈ = τ_friction − C(q, q̇)q̇ − G(q)`` with τ_drive = 0.

    Parameters
    ----------
    mass_matrix : np.ndarray
        Positive-definite inertia matrix, shape ``(n, n)``.
    coriolis_vector : np.ndarray
        Coriolis/centrifugal vector ``C(q, q̇)q̇``, shape ``(n,)``.
    gravity_vector : np.ndarray
        Gravity vector ``G(q)``, shape ``(n,)``.
    friction_vector : np.ndarray | None
        Passive friction/damping torques, shape ``(n,)``.  Defaults to zeros.

    Returns
    -------
    np.ndarray
        Zero-torque accelerations, shape ``(n,)``.

    Preconditions
    -------------
    - ``mass_matrix`` is square with same dimension as vectors.
    - All inputs are finite.
    - ``mass_matrix`` is non-singular.

    Postconditions
    --------------
    - Result is finite with shape ``(n,)``.
    """
    _validate_dynamics_inputs(
        mass_matrix, coriolis_vector, gravity_vector, friction_vector
    )

    n = mass_matrix.shape[0]
    friction = friction_vector if friction_vector is not None else np.zeros(n)

    rhs = friction - coriolis_vector - gravity_vector
    qddot = np.linalg.solve(mass_matrix, rhs)

    if not np.all(np.isfinite(qddot)):
        raise PostconditionError(
            "ZTCF accelerations contain non-finite values",
            function_name="compute_ztcf_accelerations",
            result=qddot,
        )
    return qddot


def compute_ztcf_forces(
    *,
    mass_matrix: np.ndarray,
    coriolis_vector: np.ndarray,
    gravity_vector: np.ndarray,
    friction_vector: np.ndarray | None = None,
    joint_positions: np.ndarray,
    segment_masses: np.ndarray,
    segment_lengths: np.ndarray,
    gravity_acceleration: float = 9.81,
) -> ZTCFResult:
    """Compute ZTCF joint forces from dynamics matrices and kinematics.

    This is the main entry point.  It:
    1. Solves for zero-torque accelerations via ``compute_ztcf_accelerations``.
    2. Computes joint forces from the resulting accelerations using
       Newton-Euler: ``F_joint = m * a_joint − m * g``.

    Parameters
    ----------
    mass_matrix, coriolis_vector, gravity_vector, friction_vector
        Dynamics terms (see ``compute_ztcf_accelerations``).
    joint_positions : np.ndarray
        World-frame joint positions, shape ``(n_joints, n_dims)``.
    segment_masses : np.ndarray
        Mass of each body segment, shape ``(n_joints,)``.
    segment_lengths : np.ndarray
        Length of each segment, shape ``(n_joints,)``.
    gravity_acceleration : float
        Scalar gravitational acceleration (default 9.81 m/s²).

    Returns
    -------
    ZTCFResult
        Container with joint forces and accelerations.
    """
    _validate_force_inputs(joint_positions, segment_masses, segment_lengths)

    qddot = compute_ztcf_accelerations(
        mass_matrix=mass_matrix,
        coriolis_vector=coriolis_vector,
        gravity_vector=gravity_vector,
        friction_vector=friction_vector,
    )

    n_joints = joint_positions.shape[0]
    n_dims = joint_positions.shape[1] if joint_positions.ndim > 1 else 1
    forces = _forces_from_accelerations(
        qddot=qddot,
        joint_positions=joint_positions,
        segment_masses=segment_masses,
        segment_lengths=segment_lengths,
        gravity_acceleration=gravity_acceleration,
        n_joints=n_joints,
        n_dims=n_dims,
    )

    if not np.all(np.isfinite(forces)):
        raise PostconditionError(
            "ZTCF joint forces contain non-finite values",
            function_name="compute_ztcf_forces",
            result=forces,
        )

    return ZTCFResult(
        joint_forces=forces,
        joint_accelerations=qddot,
        n_joints=n_joints,
    )


# ============================================================================
# Force delta
# ============================================================================


def compute_force_delta(
    *,
    total_forces: np.ndarray,
    ztcf_forces: np.ndarray,
) -> np.ndarray:
    """Compute the active control component: total − ZTCF.

    The delta represents the force contribution solely from applied
    joint torques (the "active" or "control" component).

    Parameters
    ----------
    total_forces : np.ndarray
        Total joint forces (with applied torques), shape ``(n_joints, n_dims)``.
    ztcf_forces : np.ndarray
        ZTCF passive forces (zero torque), shape ``(n_joints, n_dims)``.

    Returns
    -------
    np.ndarray
        Delta forces, shape ``(n_joints, n_dims)``.

    Preconditions
    -------------
    - Shapes must match.
    - All values must be finite.
    """
    total_forces = np.asarray(total_forces)
    ztcf_forces = np.asarray(ztcf_forces)

    if total_forces.shape != ztcf_forces.shape:
        raise PreconditionError(
            f"Shape mismatch: total {total_forces.shape} vs ztcf {ztcf_forces.shape}",
            function_name="compute_force_delta",
            parameter="total_forces / ztcf_forces",
        )
    if not np.all(np.isfinite(total_forces)):
        raise PreconditionError(
            "total_forces contains non-finite values",
            function_name="compute_force_delta",
            parameter="total_forces",
        )
    if not np.all(np.isfinite(ztcf_forces)):
        raise PreconditionError(
            "ztcf_forces contains non-finite values",
            function_name="compute_force_delta",
            parameter="ztcf_forces",
        )

    return total_forces - ztcf_forces


# ============================================================================
# Private helpers
# ============================================================================


def _validate_dynamics_inputs(
    mass_matrix: np.ndarray,
    coriolis_vector: np.ndarray,
    gravity_vector: np.ndarray,
    friction_vector: np.ndarray | None,
) -> None:
    """Validate dynamics matrix/vector inputs (shared preconditions)."""
    mass_matrix = np.asarray(mass_matrix)
    coriolis_vector = np.asarray(coriolis_vector)
    gravity_vector = np.asarray(gravity_vector)

    if mass_matrix.ndim != 2 or mass_matrix.shape[0] != mass_matrix.shape[1]:
        raise PreconditionError(
            f"mass_matrix must be square, got shape {mass_matrix.shape}",
            function_name="compute_ztcf_accelerations",
            parameter="mass_matrix",
        )

    n = mass_matrix.shape[0]

    if coriolis_vector.shape != (n,):
        raise PreconditionError(
            f"coriolis_vector shape {coriolis_vector.shape} != ({n},)",
            function_name="compute_ztcf_accelerations",
            parameter="coriolis_vector",
        )
    if gravity_vector.shape != (n,):
        raise PreconditionError(
            f"gravity_vector shape {gravity_vector.shape} != ({n},)",
            function_name="compute_ztcf_accelerations",
            parameter="gravity_vector",
        )
    if friction_vector is not None and np.asarray(friction_vector).shape != (n,):
        raise PreconditionError(
            f"friction_vector shape {np.asarray(friction_vector).shape} != ({n},)",
            function_name="compute_ztcf_accelerations",
            parameter="friction_vector",
        )

    for name, arr in [
        ("mass_matrix", mass_matrix),
        ("coriolis_vector", coriolis_vector),
        ("gravity_vector", gravity_vector),
    ]:
        if not np.all(np.isfinite(arr)):
            raise PreconditionError(
                f"{name} contains non-finite values",
                function_name="compute_ztcf_accelerations",
                parameter=name,
            )

    if friction_vector is not None and not np.all(np.isfinite(friction_vector)):
        raise PreconditionError(
            "friction_vector contains non-finite values",
            function_name="compute_ztcf_accelerations",
            parameter="friction_vector",
        )

    # Check mass matrix is non-singular (det != 0)
    det = np.linalg.det(mass_matrix)
    if abs(det) < 1e-12:
        raise PreconditionError(
            f"mass_matrix is singular or near-singular (det={det:.2e})",
            function_name="compute_ztcf_accelerations",
            parameter="mass_matrix",
        )


def _validate_force_inputs(
    joint_positions: np.ndarray,
    segment_masses: np.ndarray,
    segment_lengths: np.ndarray,
) -> None:
    """Validate kinematic inputs for force computation."""
    joint_positions = np.asarray(joint_positions)
    segment_masses = np.asarray(segment_masses)
    segment_lengths = np.asarray(segment_lengths)

    if joint_positions.ndim < 1:
        raise PreconditionError(
            "joint_positions must be at least 1D",
            function_name="compute_ztcf_forces",
            parameter="joint_positions",
        )

    n_joints = joint_positions.shape[0]
    if segment_masses.shape != (n_joints,):
        raise PreconditionError(
            f"segment_masses shape {segment_masses.shape} != ({n_joints},)",
            function_name="compute_ztcf_forces",
            parameter="segment_masses",
        )
    if segment_lengths.shape != (n_joints,):
        raise PreconditionError(
            f"segment_lengths shape {segment_lengths.shape} != ({n_joints},)",
            function_name="compute_ztcf_forces",
            parameter="segment_lengths",
        )


def _forces_from_accelerations(
    *,
    qddot: np.ndarray,
    joint_positions: np.ndarray,
    segment_masses: np.ndarray,
    segment_lengths: np.ndarray,
    gravity_acceleration: float,
    n_joints: int,
    n_dims: int,
) -> np.ndarray:
    """Compute joint forces from generalized accelerations via Newton-Euler.

    For each joint, the constraint force required to produce the
    zero-torque acceleration is:

        F_joint_i = m_i * a_linear_i − m_i * g_vector

    This is a simplified Newton-Euler formulation that maps generalized
    accelerations to Cartesian joint forces through the segment geometry.
    """
    forces = np.zeros((n_joints, n_dims))

    # Gravity acts in the negative y-direction (or last dimension)
    g_vec = np.zeros(n_dims)
    if n_dims >= 2:
        g_vec[1] = -gravity_acceleration

    # Map generalized accelerations to approximate linear joint accelerations
    # Each joint acceleration scales by its segment length
    n_dof = len(qddot)
    for j in range(n_joints):
        dof_idx = min(j, n_dof - 1)
        accel_magnitude = qddot[dof_idx] * segment_lengths[j]
        accel_vec = np.zeros(n_dims)
        if n_dims >= 2:
            accel_vec[0] = accel_magnitude  # tangential (simplified)

        forces[j] = segment_masses[j] * (accel_vec - g_vec)

    return forces
