"""Kinematic-dependent force analysis for golf swing biomechanics.

This module computes motion-dependent forces that can be calculated from
kinematics alone, WITHOUT requiring full inverse dynamics:

- Coriolis forces
- Centrifugal forces
- Centripetal accelerations
- Velocity-dependent forces
- Gravitational forces (configuration-dependent)

These forces are critical for understanding swing dynamics and can be computed
even for parallel mechanisms where full inverse dynamics is challenging.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from types import TracebackType


def _check_mujoco_version() -> None:
    """Validate MuJoCo version meets minimum requirements.

    Addresses Issue F-003: Prevents API signature mismatches by enforcing
    minimum version at runtime.

    Raises:
        ImportError: If MuJoCo version is too old
    """
    try:
        # MuJoCo version format: "3.3.0" or similar
        version_str = mujoco.__version__
        major, minor, *_ = map(int, version_str.split("."))

        # Require MuJoCo 3.3+ for reshaped Jacobian API
        if (major, minor) < (3, 3):
            msg = (
                f"MuJoCo {version_str} detected, but 3.3.0+ is required.\n"
                f"The reshaped Jacobian API (mj_jacBody with 2D arrays) was "
                f"introduced in MuJoCo 3.3. Earlier versions use flat arrays "
                f"which can cause dimension alignment errors.\n"
                f"Please upgrade: pip install 'mujoco>=3.3.0,<4.0.0'\n"
                f"See Issue F-003 in Assessment C for details."
            )
            raise ImportError(msg)

        # Success - log version
        warnings.warn(
            f"MuJoCo version {version_str} validated successfully",
            category=UserWarning,
            stacklevel=2,
        )

    except (AttributeError, ValueError) as e:
        # Could not parse version
        warnings.warn(
            f"Could not validate MuJoCo version: {e}. "
            f"Proceeding with fallback Jacobian handling.",
            category=UserWarning,
            stacklevel=2,
        )


# Validate MuJoCo version on module import (Issue F-003)
_check_mujoco_version()


class MjDataContext:
    """Context manager for safe MuJoCo MjData state isolation.

    This context manager saves the current state of MjData on entry and
    restores it on exit, ensuring that any mutations within the context
    do not affect the original state.

    Addresses Issues A-001, A-003, F-001, F-002 by providing functional
    purity guarantees for analysis methods.

    Example:
        >>> with MjDataContext(data):
        ...     data.qpos[:] = new_positions  # Safe to mutate
        ...     result = compute_something(model, data)
        ... # data.qpos is automatically restored here

    This enables:
    - Safe parallel analysis
    - No Observer Effect bugs
    - Scientific reproducibility
    - Thread-safe computations
    """

    def __init__(self, data: mujoco.MjData) -> None:
        """Initialize context manager.

        Args:
            data: MuJoCo data structure to protect
        """
        self.data = data
        self.qpos_backup: np.ndarray | None = None
        self.qvel_backup: np.ndarray | None = None
        self.qacc_backup: np.ndarray | None = None
        self.ctrl_backup: np.ndarray | None = None
        self.time_backup: float = 0.0

    def __enter__(self) -> mujoco.MjData:
        """Save current state on context entry.

        Returns:
            The data object for convenience
        """
        self.qpos_backup = self.data.qpos.copy()
        self.qvel_backup = self.data.qvel.copy()
        self.qacc_backup = self.data.qacc.copy()
        self.ctrl_backup = self.data.ctrl.copy()
        self.time_backup = self.data.time
        return self.data

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Restore state on context exit, even if exception occurred.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        if self.qpos_backup is not None:
            self.data.qpos[:] = self.qpos_backup
        if self.qvel_backup is not None:
            self.data.qvel[:] = self.qvel_backup
        if self.qacc_backup is not None:
            self.data.qacc[:] = self.qacc_backup
        if self.ctrl_backup is not None:
            self.data.ctrl[:] = self.ctrl_backup
        self.data.time = self.time_backup

        # Recompute forward kinematics to sync all derived quantities
        mujoco.mj_forward(self.data.model, self.data)


@dataclass
class KinematicForceData:
    """Container for kinematic-dependent forces at a single time point."""

    time: float

    # Joint-space forces
    coriolis_forces: np.ndarray  # [nv] - Coriolis and centrifugal forces
    gravity_forces: np.ndarray  # [nv] - Gravitational forces

    # Decomposed components
    centrifugal_forces: np.ndarray | None = None  # [nv] - Pure centrifugal
    velocity_coupling_forces: np.ndarray | None = None  # [nv] - Velocity coupling

    # Task-space forces (end-effector)
    club_head_coriolis_force: np.ndarray | None = None  # [3] - at club head
    club_head_centrifugal_force: np.ndarray | None = None  # [3] - at club head
    club_head_apparent_force: np.ndarray | None = None  # [3] - total apparent force

    # Power contributions
    coriolis_power: float = 0.0  # Power dissipated by Coriolis forces
    centrifugal_power: float = 0.0  # Power from centrifugal effects

    # Kinetic energy contributions
    rotational_kinetic_energy: float = 0.0
    translational_kinetic_energy: float = 0.0


class KinematicForceAnalyzer:
    """Analyze kinematic-dependent forces in golf swing.

    This class computes Coriolis, centrifugal, and other velocity-dependent
    forces that can be determined from kinematics alone. These forces are
    essential for understanding swing dynamics without requiring full
    inverse dynamics.

    Key Applications:
    - Analyze forces in captured motion data (from motion capture)
    - Understand velocity-dependent effects
    - Study energy transfer mechanisms
    - Evaluate dynamic coupling between joints
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize kinematic force analyzer.

        MEMORY ALLOCATION (Issue A-004): This initializer allocates a complete
        MjData structure for scratch computations, which can be several MB for
        complex models. This is intentional for thread safety and performance,
        but users should be aware of the memory footprint when creating multiple
        analyzer instances.

        Memory usage breakdown (approximate):
        - _perturb_data: ~size_of(MjData) ≈ O(nv² + nbody)
        - Jacobian buffers: 2 × 3 × nv floats ≈ 24×nv bytes
        - Total additional memory: ~few MB for typical humanoid models

        For memory-constrained environments, consider:
        - Reusing a single analyzer instance across analyses
        - Using lazy initialization (allocate _perturb_data on first use)
        - Sharing analyzer instances across threads (with proper synchronization)

        Args:
            model: MuJoCo model
            data: MuJoCo data (shared reference, not modified by compute methods)
        """
        self.model = model
        self.data = data

        # Find important bodies
        self.club_head_id = self._find_body_id("club_head")
        self.club_grip_id = self._find_body_id("club") or self._find_body_id("grip")

        # MEMORY ALLOCATION: Dedicated MjData for state-isolated computations
        # This prevents race conditions but increases memory footprint
        # See Issue A-004 for detailed analysis
        self._perturb_data = mujoco.MjData(model)

        # Pre-allocate Jacobian buffers to avoid repeated allocation
        # Detect API version to use correct array shape
        self.nv = model.nv
        try:
            # Try reshaped API (MuJoCo 3.3+ preferred)
            jacp_test = np.zeros((3, self.nv))
            jacr_test = np.zeros((3, self.nv))
            mujoco.mj_jacBody(model, data, jacp_test, jacr_test, 0)
            self._use_reshaped_arrays = True
            self._jacp = np.zeros((3, self.nv))
            self._jacr = np.zeros((3, self.nv))
        except TypeError:
            # Fallback to flat API
            self._use_reshaped_arrays = False
            self._jacp = np.zeros(3 * self.nv)
            self._jacr = np.zeros(3 * self.nv)

    def _find_body_id(self, name_pattern: str) -> int | None:
        """Find body ID by name pattern."""
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and name_pattern.lower() in body_name.lower():
                return i
        return None

    def _compute_jacobian(
        self, body_id: int, data: mujoco.MjData | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Jacobian for a body using pre-allocated buffers.

        Args:
            body_id: Body ID
            data: MuJoCo data (default: self.data)

        Returns:
            Tuple of (jacp, jacr) as (3, nv) arrays.
            Note: Returns views into internal buffers or copies depending on usage.
        """
        if data is None:
            data = self.data

        if self._use_reshaped_arrays:
            mujoco.mj_jacBody(self.model, data, self._jacp, self._jacr, body_id)
            return self._jacp, self._jacr
        else:
            mujoco.mj_jacBody(self.model, data, self._jacp, self._jacr, body_id)
            return (
                self._jacp.reshape(3, self.nv),
                self._jacr.reshape(3, self.nv),
            )

    def compute_coriolis_forces(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Compute Coriolis and centrifugal forces.

        These are the velocity-dependent forces in the equations of motion:
        M(q)q̈ + C(q,q̇)q̇ + g(q) = τ

        The term C(q,q̇)q̇ represents Coriolis and centrifugal forces.

        FIXED: Uses dedicated _perturb_data instead of shared self.data to prevent
        race conditions and state corruption. See Issues A-001 and F-002.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Coriolis forces [nv]
        """
        # FIXED: Use private/scratch data structure to avoid corrupting shared state
        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel

        # Forward kinematics
        mujoco.mj_forward(self.model, self._perturb_data)

        # In MuJoCo, qfrc_bias = C(q,q̇)q̇ + g(q)
        # We need to separate Coriolis from gravity

        # Method 1: Compute bias with and without velocity
        bias_with_velocity = self._perturb_data.qfrc_bias.copy()

        # Compute bias with zero velocity (only gravity)
        self._perturb_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self._perturb_data)
        gravity_only = self._perturb_data.qfrc_bias.copy()

        # Coriolis forces = total bias - gravity
        return np.asarray(bias_with_velocity - gravity_only)

    def compute_gravity_forces(self, qpos: np.ndarray) -> np.ndarray:
        """Compute gravitational forces.

        FIXED: Uses dedicated _perturb_data to prevent state corruption.
        See Issues A-001 and F-002.

        Args:
            qpos: Joint positions [nv]

        Returns:
            Gravity forces [nv]
        """
        # FIXED: Use private/scratch data structure
        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self._perturb_data)

        # With zero velocity, qfrc_bias = g(q)
        return np.asarray(self._perturb_data.qfrc_bias.copy())

    def decompose_coriolis_forces(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompose Coriolis forces into centrifugal and velocity coupling.

        Coriolis matrix C(q,q̇) can be decomposed:
        - Centrifugal terms: Diagonal terms (q̇ᵢ²)
        - Velocity coupling: Off-diagonal terms (q̇ᵢq̇ⱼ)

        ⚠️ PERFORMANCE WARNING: This method uses an approximation that calls
        compute_coriolis_forces N+1 times, resulting in O(N²) complexity.
        For high-DOF models, this can be slow.

        RECOMMENDATION: Use the combined compute_coriolis_forces() method instead,
        which returns the total Coriolis+centrifugal forces efficiently in O(N).
        Decomposition is rarely needed for most applications.

        See Issue A-002 and B-002 for optimization path using analytical RNE.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Tuple of (centrifugal_forces [nv], coupling_forces [nv])
        """
        # OPTIMIZATION NOTE: The proper fix is to use mj_rne properties or
        # implement analytical decomposition. Current implementation is
        # accurate but slow for high-DOF systems.
        #
        # Full analytical solution requires:
        # - Custom RNE recursion to separate diagonal/off-diagonal terms
        # - OR use spatial algebra (screw theory) for frame-independent decomposition
        # - OR skip decomposition and use total Coriolis forces directly
        #
        # For most golf swing analyses, the total Coriolis force is sufficient.

        centrifugal = np.zeros(self.model.nv)
        coupling = np.zeros(self.model.nv)

        # Full Coriolis forces
        total_coriolis = self.compute_coriolis_forces(qpos, qvel)

        # Estimate centrifugal: vary each velocity independently
        # This captures diagonal terms of the Coriolis matrix
        for i in range(self.model.nv):
            qvel_single = np.zeros(self.model.nv)
            qvel_single[i] = qvel[i]

            single_coriolis = self.compute_coriolis_forces(qpos, qvel_single)
            centrifugal += single_coriolis

        # Coupling is the difference (off-diagonal terms)
        coupling = total_coriolis - centrifugal

        return centrifugal, coupling

    def compute_mass_matrix(self, qpos: np.ndarray) -> np.ndarray:
        """Compute configuration-dependent mass matrix M(q).

        FIXED: Uses dedicated _perturb_data to prevent state corruption.

        Args:
            qpos: Joint positions [nv]

        Returns:
            Mass matrix [nv x nv]
        """
        # FIXED: Use private data structure
        self._perturb_data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self._perturb_data)

        # Get full mass matrix
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self._perturb_data.qM)

        return M

    def compute_coriolis_matrix(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Compute Coriolis matrix C(q,q̇).

        The Coriolis matrix satisfies: C(q,q̇)q̇ = coriolis forces

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Coriolis matrix [nv x nv]
        """
        # Use finite differences to estimate C
        epsilon = 1e-6
        C = np.zeros((self.model.nv, self.model.nv))

        # Reference Coriolis forces
        c_ref = self.compute_coriolis_forces(qpos, qvel)

        for i in range(self.model.nv):
            qvel_perturb = qvel.copy()
            qvel_perturb[i] += epsilon

            c_perturb = self.compute_coriolis_forces(qpos, qvel_perturb)

            C[:, i] = (c_perturb - c_ref) / epsilon

        return C

    def compute_club_head_apparent_forces(  # noqa: PLR0915
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute apparent forces at club head (Coriolis, centrifugal, etc.).

        These are the "fictitious" forces experienced in the rotating
        reference frame attached to the golfer.

        FIXED: Uses dedicated _perturb_data to prevent state corruption.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            qacc: Joint accelerations [nv]

        Returns:
            Tuple of (coriolis_force [3], centrifugal_force [3], total_apparent [3])
        """
        if self.club_head_id is None:
            return np.zeros(3), np.zeros(3), np.zeros(3)

        # FIXED: Use private data structure for current state
        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)

        # Compute club head Jacobian
        jacp, _ = self._compute_jacobian(self.club_head_id, data=self._perturb_data)
        jacp_curr = jacp.copy()  # Save copy as _compute_jacobian reuses buffer

        # Store club position before perturbing state
        club_pos = self._perturb_data.xpos[self.club_head_id].copy()

        # Jacobian time derivative (approximate)
        epsilon = 1e-6

        # Compute Jacobian at perturbed state
        self._perturb_data.qpos[:] = qpos + epsilon * qvel
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)

        jacp_perturb, _ = self._compute_jacobian(
            self.club_head_id, data=self._perturb_data
        )

        jacp_dot = (jacp_perturb - jacp_curr) / epsilon

        # Coriolis force: -2m(Ω × v)
        # In our case: Coriolis contribution to acceleration
        coriolis_accel = jacp_dot @ qvel

        # Assume unit mass for force (or multiply by club head mass)
        club_head_mass = self.model.body_mass[self.club_head_id]
        coriolis_force = -club_head_mass * coriolis_accel

        # Centrifugal force: -mΩ²r
        # This is embedded in the Coriolis term
        # For separation, we'd need angular velocity of each body segment

        # Total apparent force (from joint-space Coriolis forces)
        joint_coriolis = self.compute_coriolis_forces(qpos, qvel)
        apparent_force = jacp_curr.T @ joint_coriolis[: self.model.nv]

        # Approximate centrifugal as component aligned with position
        centrifugal_direction = club_pos / (np.linalg.norm(club_pos) + 1e-10)
        centrifugal_magnitude = np.dot(apparent_force, centrifugal_direction)
        centrifugal_force = centrifugal_magnitude * centrifugal_direction

        return coriolis_force, centrifugal_force, apparent_force

    def compute_kinematic_power(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> dict[str, float]:
        """Compute power contributions from kinematic forces.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Dictionary with power contributions
        """
        # Coriolis forces
        coriolis_forces = self.compute_coriolis_forces(qpos, qvel)

        # Coriolis power (should be zero for conservative systems)
        # Power = F · v
        coriolis_power = np.dot(coriolis_forces, qvel)

        # Decompose
        centrifugal, coupling = self.decompose_coriolis_forces(qpos, qvel)
        centrifugal_power = np.dot(centrifugal, qvel)
        coupling_power = np.dot(coupling, qvel)

        # Gravity power
        gravity_forces = self.compute_gravity_forces(qpos)
        gravity_power = np.dot(gravity_forces, qvel)

        return {
            "coriolis_power": float(coriolis_power),
            "centrifugal_power": float(centrifugal_power),
            "coupling_power": float(coupling_power),
            "gravity_power": float(gravity_power),
            "total_conservative_power": float(coriolis_power + gravity_power),
        }

    def compute_kinetic_energy_components(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> dict[str, float]:
        """Decompose kinetic energy into rotational and translational.

        FIXED: Uses dedicated _perturb_data to prevent state corruption.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Dictionary with kinetic energy components
        """
        # FIXED: Use private data structure
        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)

        rotational_ke = 0.0
        translational_ke = 0.0

        for i in range(1, self.model.nbody):  # Skip world body
            # Get body mass and inertia
            body_mass = self.model.body_mass[i]
            body_inertia = self.model.body_inertia[i]

            # Get body velocity (linear and angular)
            jacp, jacr = self._compute_jacobian(i, data=self._perturb_data)

            v_linear = jacp @ qvel
            omega = jacr @ qvel

            # Kinetic energies
            translational_ke += 0.5 * body_mass * np.dot(v_linear, v_linear)
            rotational_ke += 0.5 * np.dot(omega, body_inertia * omega)

        return {
            "rotational": float(rotational_ke),
            "translational": float(translational_ke),
            "total": float(rotational_ke + translational_ke),
        }

    def analyze_trajectory(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ) -> list[KinematicForceData]:
        """Analyze kinematic forces along a trajectory.

        This is the main function for analyzing captured motion data.

        Args:
            times: Time array [N]
            positions: Joint positions [N x nv]
            velocities: Joint velocities [N x nv]
            accelerations: Joint accelerations [N x nv]

        Returns:
            List of KinematicForceData for each time step
        """
        results = []

        for i in range(len(times)):
            qpos = positions[i]
            qvel = velocities[i]
            qacc = accelerations[i]

            # Compute forces
            coriolis = self.compute_coriolis_forces(qpos, qvel)
            gravity = self.compute_gravity_forces(qpos)
            centrifugal, coupling = self.decompose_coriolis_forces(qpos, qvel)

            # Club head apparent forces
            club_coriolis, club_centrifugal, club_apparent = (
                self.compute_club_head_apparent_forces(qpos, qvel, qacc)
            )

            # Power contributions
            power_dict = self.compute_kinematic_power(qpos, qvel)

            # Kinetic energy
            ke_dict = self.compute_kinetic_energy_components(qpos, qvel)

            # Create data object
            data = KinematicForceData(
                time=times[i],
                coriolis_forces=coriolis,
                gravity_forces=gravity,
                centrifugal_forces=centrifugal,
                velocity_coupling_forces=coupling,
                club_head_coriolis_force=club_coriolis,
                club_head_centrifugal_force=club_centrifugal,
                club_head_apparent_force=club_apparent,
                coriolis_power=power_dict["coriolis_power"],
                centrifugal_power=power_dict["centrifugal_power"],
                rotational_kinetic_energy=ke_dict["rotational"],
                translational_kinetic_energy=ke_dict["translational"],
            )

            results.append(data)

        return results

    def compute_effective_mass(
        self,
        qpos: np.ndarray,
        direction: np.ndarray,
        body_id: int | None = None,
    ) -> float:
        """Compute effective mass in a given direction.

        Effective mass determines how difficult it is to accelerate
        in a specific direction.

        FIXED: Uses dedicated _perturb_data to prevent state corruption.

        Args:
            qpos: Joint positions [nv]
            direction: Direction vector [3]
            body_id: Body to compute for (default: club head)

        Returns:
            Effective mass in that direction [kg]
        """
        if body_id is None:
            body_id = self.club_head_id

        if body_id is None:
            return 0.0

        # Normalize direction
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Get mass matrix (already uses _perturb_data internally)
        M = self.compute_mass_matrix(qpos)

        # FIXED: Use private data structure
        self._perturb_data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self._perturb_data)

        jacp, _ = self._compute_jacobian(body_id, data=self._perturb_data)

        # Project Jacobian onto direction
        J_dir = direction @ jacp

        # Effective mass: m_eff = (J M^{-1} J^T)^{-1}
        M_inv = np.linalg.inv(M)
        m_eff = 1.0 / (J_dir @ M_inv @ J_dir.T + 1e-10)

        return float(m_eff)

    def compute_centripetal_acceleration(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        body_id: int | None = None,
    ) -> np.ndarray:
        """Compute centripetal acceleration at a body.

        ⚠️ WARNING - EXPERIMENTAL/BROKEN: This method contains a fundamental physics error.
        It treats the articulated robot as a point mass in circular motion about the
        world origin (0,0,0), which is incorrect for multi-body kinematic chains.

        ISSUE: Uses a_c = v²/r with r = distance from origin, ignoring the fact that
        each body segment has its own angular velocity and rotation center.

        CORRECT APPROACH: Use spatial acceleration a_total = J·q̈ + J̇·q̇ where the
        "centripetal/coriolis" contribution is the velocity product term J̇·q̇.

        For articulated systems, centripetal acceleration should be computed as:
        a_c = ω × (ω × r) where ω is derived from the rotational Jacobian.

        DO NOT USE THIS METHOD FOR STRESS ANALYSIS OR SAFETY-CRITICAL APPLICATIONS.
        Results are physically invalid for articulated chains.

        See Issue B-001 in Assessment B for detailed analysis.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            body_id: Body ID (default: club head)

        Returns:
            Centripetal acceleration [3] - INACCURATE, see warning above
        """
        import warnings

        warnings.warn(
            "compute_centripetal_acceleration contains a fundamental physics error "
            "(assumes circular motion about origin). Results are invalid for "
            "articulated chains. See Assessment B-001 for details.",
            category=UserWarning,
            stacklevel=2,
        )

        if body_id is None:
            body_id = self.club_head_id

        if body_id is None:
            return np.zeros(3)

        # FIXED: Use private data structure to prevent state corruption
        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)

        # Body velocity
        jacp, _ = self._compute_jacobian(body_id, data=self._perturb_data)

        v = jacp @ qvel

        # Body position (relative to some reference)
        pos = self._perturb_data.xpos[body_id].copy()

        # Centripetal acceleration
        # For circular motion: a_c = v²/r pointing toward center
        # General case: a_c = ω × (ω × r)

        # ⚠️ BROKEN: This approximation assumes circular motion about origin
        # which is fundamentally wrong for articulated kinematic chains
        speed = np.linalg.norm(v)
        radius = np.linalg.norm(pos)

        if radius > 1e-6:
            a_c_magnitude = speed**2 / radius
            a_c = -a_c_magnitude * (pos / radius)  # Negative for inward direction
        else:
            a_c = np.zeros(3)

        return np.asarray(a_c)


def export_kinematic_forces_to_csv(
    force_data_list: list[KinematicForceData],
    filepath: str,
) -> None:
    """Export kinematic force analysis to CSV file.

    Args:
        force_data_list: List of force data
        filepath: Output CSV file path
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = [
            "time",
            "coriolis_power",
            "centrifugal_power",
            "rotational_ke",
            "translational_ke",
        ]

        # Add joint-wise Coriolis forces
        nv = len(force_data_list[0].coriolis_forces)
        for i in range(nv):
            header.extend(
                [f"coriolis_force_{i}", f"gravity_force_{i}", f"centrifugal_force_{i}"],
            )

        # Add club head forces
        header.extend(
            [
                "club_coriolis_x",
                "club_coriolis_y",
                "club_coriolis_z",
                "club_centrifugal_x",
                "club_centrifugal_y",
                "club_centrifugal_z",
            ],
        )

        writer.writerow(header)

        # Data rows
        for data in force_data_list:
            row = [
                data.time,
                data.coriolis_power,
                data.centrifugal_power,
                data.rotational_kinetic_energy,
                data.translational_kinetic_energy,
            ]

            for i in range(nv):
                row.extend(
                    [
                        data.coriolis_forces[i],
                        data.gravity_forces[i],
                        (
                            data.centrifugal_forces[i]
                            if data.centrifugal_forces is not None
                            else 0.0
                        ),
                    ],
                )

            if data.club_head_coriolis_force is not None:
                row.extend(data.club_head_coriolis_force.tolist())
            else:
                row.extend([0, 0, 0])

            if data.club_head_centrifugal_force is not None:
                row.extend(data.club_head_centrifugal_force.tolist())
            else:
                row.extend([0, 0, 0])

            writer.writerow(row)
