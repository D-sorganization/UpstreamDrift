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
from dataclasses import dataclass

import mujoco
import numpy as np


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

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data

        # Find important bodies
        self.club_head_id = self._find_body_id("club_head")
        self.club_grip_id = self._find_body_id("club") or self._find_body_id("grip")

    def _find_body_id(self, name_pattern: str) -> int | None:
        """Find body ID by name pattern."""
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and name_pattern.lower() in body_name.lower():
                return i
        return None

    def compute_coriolis_forces(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Compute Coriolis and centrifugal forces.

        These are the velocity-dependent forces in the equations of motion:
        M(q)q̈ + C(q,q̇)q̇ + g(q) = τ

        The term C(q,q̇)q̇ represents Coriolis and centrifugal forces.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Coriolis forces [nv]
        """
        # Set state
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # In MuJoCo, qfrc_bias = C(q,q̇)q̇ + g(q)
        # We need to separate Coriolis from gravity

        # Method 1: Compute bias with and without velocity
        bias_with_velocity = self.data.qfrc_bias.copy()

        # Compute bias with zero velocity (only gravity)
        qvel_backup = self.data.qvel.copy()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        gravity_only = self.data.qfrc_bias.copy()

        # Restore velocity
        self.data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.model, self.data)

        # Coriolis forces = total bias - gravity
        return bias_with_velocity - gravity_only

    def compute_gravity_forces(self, qpos: np.ndarray) -> np.ndarray:
        """Compute gravitational forces.

        Args:
            qpos: Joint positions [nv]

        Returns:
            Gravity forces [nv]
        """
        # Set state with zero velocity
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # With zero velocity, qfrc_bias = g(q)
        return self.data.qfrc_bias.copy()

    def decompose_coriolis_forces(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompose Coriolis forces into centrifugal and velocity coupling.

        Coriolis matrix C(q,q̇) can be decomposed:
        - Centrifugal terms: Diagonal terms (q̇ᵢ²)
        - Velocity coupling: Off-diagonal terms (q̇ᵢq̇ⱼ)

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Tuple of (centrifugal_forces [nv], coupling_forces [nv])
        """
        # Approximate decomposition using finite differences

        centrifugal = np.zeros(self.model.nv)
        coupling = np.zeros(self.model.nv)

        # Full Coriolis forces
        total_coriolis = self.compute_coriolis_forces(qpos, qvel)

        # Estimate centrifugal: vary each velocity independently
        for i in range(self.model.nv):
            qvel_single = np.zeros(self.model.nv)
            qvel_single[i] = qvel[i]

            single_coriolis = self.compute_coriolis_forces(qpos, qvel_single)
            centrifugal += single_coriolis

        # Coupling is the difference
        coupling = total_coriolis - centrifugal

        return centrifugal, coupling

    def compute_mass_matrix(self, qpos: np.ndarray) -> np.ndarray:
        """Compute configuration-dependent mass matrix M(q).

        Args:
            qpos: Joint positions [nv]

        Returns:
            Mass matrix [nv x nv]
        """
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        # Get full mass matrix
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)

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

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            qacc: Joint accelerations [nv]

        Returns:
            Tuple of (coriolis_force [3], centrifugal_force [3], total_apparent [3])
        """
        if self.club_head_id is None:
            return np.zeros(3), np.zeros(3), np.zeros(3)

        # Set state
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        # Compute club head Jacobian
        # MuJoCo 3.3+ may require reshaped arrays
        try:
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.club_head_id)
        except TypeError:
            # Fallback to flat array approach
            jacp_flat = np.zeros(3 * self.model.nv)
            jacr_flat = np.zeros(3 * self.model.nv)
            mujoco.mj_jacBody(
                self.model, self.data, jacp_flat, jacr_flat, self.club_head_id
            )
            jacp = jacp_flat.reshape(3, self.model.nv)
            jacr = jacr_flat.reshape(3, self.model.nv)

        # Jacobian time derivative (approximate)
        epsilon = 1e-6
        jacp_dot = np.zeros((3, self.model.nv))

        # Compute Jacobian at perturbed state
        data_copy = mujoco.MjData(self.model)
        data_copy.qpos[:] = qpos + epsilon * qvel
        data_copy.qvel[:] = qvel
        mujoco.mj_forward(self.model, data_copy)

        # MuJoCo 3.3+ may require reshaped arrays
        try:
            jacp_perturb = np.zeros((3, self.model.nv))
            jacr_perturb = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(
                self.model,
                data_copy,
                jacp_perturb,
                jacr_perturb,
                self.club_head_id,
            )
        except TypeError:
            # Fallback to flat array approach
            jacp_perturb_flat = np.zeros(3 * self.model.nv)
            jacr_perturb_flat = np.zeros(3 * self.model.nv)
            mujoco.mj_jacBody(
                self.model,
                data_copy,
                jacp_perturb_flat,
                jacr_perturb_flat,
                self.club_head_id,
            )
            jacp_perturb = jacp_perturb_flat.reshape(3, self.model.nv)
            jacr_perturb = jacr_perturb_flat.reshape(3, self.model.nv)

        jacp_dot = (jacp_perturb - jacp) / epsilon

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
        apparent_force = jacp.T @ joint_coriolis[: self.model.nv]

        # Approximate centrifugal as component aligned with position
        club_pos = self.data.xpos[self.club_head_id].copy()
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

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Dictionary with kinetic energy components
        """
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        rotational_ke = 0.0
        translational_ke = 0.0

        for i in range(1, self.model.nbody):  # Skip world body
            # Get body mass and inertia
            body_mass = self.model.body_mass[i]
            body_inertia = self.model.body_inertia[i]

            # Get body velocity (linear and angular)
            # MuJoCo 3.3+ may require reshaped arrays
            try:
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, i)
            except TypeError:
                # Fallback to flat array approach
                jacp_flat = np.zeros(3 * self.model.nv)
                jacr_flat = np.zeros(3 * self.model.nv)
                mujoco.mj_jacBody(self.model, self.data, jacp_flat, jacr_flat, i)
                jacp = jacp_flat.reshape(3, self.model.nv)
                jacr = jacr_flat.reshape(3, self.model.nv)

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

        # Get mass matrix
        M = self.compute_mass_matrix(qpos)

        # Get Jacobian
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        # MuJoCo 3.3+ may require reshaped arrays
        try:
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        except TypeError:
            # Fallback to flat array approach
            jacp_flat = np.zeros(3 * self.model.nv)
            jacr_flat = np.zeros(3 * self.model.nv)
            mujoco.mj_jacBody(self.model, self.data, jacp_flat, jacr_flat, body_id)
            jacp = jacp_flat.reshape(3, self.model.nv)
            jacr = jacr_flat.reshape(3, self.model.nv)

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

        Centripetal acceleration = v²/r pointing toward center of rotation

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            body_id: Body ID (default: club head)

        Returns:
            Centripetal acceleration [3]
        """
        if body_id is None:
            body_id = self.club_head_id

        if body_id is None:
            return np.zeros(3)

        # Get body state
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        # Body velocity
        # MuJoCo 3.3+ may require reshaped arrays
        try:
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        except TypeError:
            # Fallback to flat array approach
            jacp_flat = np.zeros(3 * self.model.nv)
            jacr_flat = np.zeros(3 * self.model.nv)
            mujoco.mj_jacBody(self.model, self.data, jacp_flat, jacr_flat, body_id)
            jacp = jacp_flat.reshape(3, self.model.nv)
            jacr = jacr_flat.reshape(3, self.model.nv)

        v = jacp @ qvel

        # Body position (relative to some reference)
        pos = self.data.xpos[body_id].copy()

        # Centripetal acceleration
        # For circular motion: a_c = v²/r pointing toward center
        # General case: a_c = ω × (ω × r)

        # Approximate: Use velocity squared divided by distance
        speed = np.linalg.norm(v)
        radius = np.linalg.norm(pos)

        if radius > 1e-6:
            a_c_magnitude = speed**2 / radius
            a_c = -a_c_magnitude * (pos / radius)  # Negative for inward direction
        else:
            a_c = np.zeros(3)

        return a_c


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
