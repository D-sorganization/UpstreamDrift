from __future__ import annotations

import mujoco
import numpy as np

from src.shared.python.core.numerical_constants import (
    EPSILON_FINITE_DIFF_JACOBIAN,
    EPSILON_SINGULARITY_DETECTION,
)

from ..jacobian_utils import (
    check_jacobian_rank,
    check_mass_matrix_conditioning,
    compute_coriolis_matrix,
    compute_effective_mass_value,
    compute_jacobian,
    compute_mass_matrix,
    validate_effective_mass_direction,
)
from .types import KinematicForceData


class KinematicForceAnalyzer:
    """Analyze kinematic-dependent forces in golf swing.

    This class computes Coriolis, centrifugal, and other velocity-dependent
    forces that can be determined from kinematics alone. These forces are
    essential for understanding swing dynamics without requiring full
    inverse dynamics.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize kinematic force analyzer.

        Args:
            model: MuJoCo model
            data: MuJoCo data (shared reference, not modified by compute methods)
        """
        if model is None:
            raise ValueError("model must be provided")
        self.model = model
        self.data = data

        # Find important bodies
        self.club_head_id = self._find_body_id("club_head")
        self.club_grip_id = self._find_body_id("club") or self._find_body_id("grip")

        self._perturb_data = mujoco.MjData(model)

        self.nv = model.nv
        try:
            jacp_test = np.zeros((3, self.nv))
            jacr_test = np.zeros((3, self.nv))
            mujoco.mj_jacBody(model, data, jacp_test, jacr_test, 0)
            self._use_reshaped_arrays = True
            self._jacp = np.zeros((3, self.nv))
            self._jacr = np.zeros((3, self.nv))
        except TypeError:
            self._use_reshaped_arrays = False
            self._jacp = np.zeros(3 * self.nv)
            self._jacr = np.zeros(3 * self.nv)

    def _find_body_id(self, name_pattern: str) -> int | None:
        """Find body ID by name pattern."""
        if name_pattern is None:
            raise ValueError("name_pattern must be provided")
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
        """
        if data is None:
            data = self.data
        return compute_jacobian(
            self.model,
            data,
            body_id,
            self._jacp,
            self._jacr,
            self._use_reshaped_arrays,
            self.nv,
        )

    def compute_coriolis_forces(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Compute Coriolis and centrifugal forces.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Coriolis forces [nv]
        """
        return self.compute_coriolis_forces_rne(qpos, qvel)

    def compute_coriolis_forces_rne(
        self, qpos: np.ndarray, qvel: np.ndarray
    ) -> np.ndarray:
        """Compute Coriolis forces using analytical RNE.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Coriolis forces [nv]
        """
        if qpos is None:
            raise ValueError("qpos must be provided")
        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel
        self._perturb_data.qacc[:] = 0.0

        bias = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, self._perturb_data, 0, bias)

        self._perturb_data.qvel[:] = 0.0
        gravity = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, self._perturb_data, 0, gravity)

        return bias - gravity

    def compute_gravity_forces(self, qpos: np.ndarray) -> np.ndarray:
        """Compute gravitational forces.

        Args:
            qpos: Joint positions [nv]

        Returns:
            Gravity forces [nv]
        """
        if qpos is None:
            raise ValueError("qpos must be provided")
        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self._perturb_data)
        return np.asarray(self._perturb_data.qfrc_bias.copy())

    def decompose_coriolis_forces(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompose Coriolis forces into centrifugal and velocity coupling.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Tuple of (centrifugal_forces [nv], coupling_forces [nv])
        """
        if qpos is None:
            raise ValueError("qpos must be provided")
        centrifugal = np.zeros(self.model.nv)
        total_coriolis = self.compute_coriolis_forces(qpos, qvel)

        for i in range(self.model.nv):
            qvel_single = np.zeros(self.model.nv)
            qvel_single[i] = qvel[i]
            single_coriolis = self.compute_coriolis_forces(qpos, qvel_single)
            centrifugal += single_coriolis

        coupling = total_coriolis - centrifugal
        return centrifugal, coupling

    def compute_mass_matrix(self, qpos: np.ndarray) -> np.ndarray:
        """Compute configuration-dependent mass matrix M(q).

        Args:
            qpos: Joint positions [nv]

        Returns:
            Mass matrix [nv x nv]
        """
        return compute_mass_matrix(self.model, self._perturb_data, qpos)

    def compute_coriolis_matrix(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Compute Coriolis matrix C(q,q̇).

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Coriolis matrix [nv x nv]
        """
        return compute_coriolis_matrix(
            self.model, qpos, qvel, self.compute_coriolis_forces
        )

    def compute_club_head_apparent_forces(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute apparent forces at club head.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            qacc: Joint accelerations [nv]

        Returns:
            Tuple of (coriolis_force [3], centrifugal_force [3], total_apparent [3])
        """
        if qpos is None:
            raise ValueError("qpos must be provided")
        if self.club_head_id is None:
            return np.zeros(3), np.zeros(3), np.zeros(3)

        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)

        jacp, _ = self._compute_jacobian(self.club_head_id, data=self._perturb_data)
        jacp_curr = jacp.copy()

        club_pos = self._perturb_data.xpos[self.club_head_id].copy()
        epsilon = EPSILON_FINITE_DIFF_JACOBIAN

        self._perturb_data.qpos[:] = qpos + epsilon * qvel
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)
        jacp_forward, _ = self._compute_jacobian(
            self.club_head_id, data=self._perturb_data
        )
        jacp_forward = jacp_forward.copy()

        self._perturb_data.qpos[:] = qpos - epsilon * qvel
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)
        jacp_backward, _ = self._compute_jacobian(
            self.club_head_id, data=self._perturb_data
        )

        jacp_dot = (jacp_forward - jacp_backward) / (2.0 * epsilon)
        coriolis_accel = jacp_dot @ qvel

        club_head_mass = self.model.body_mass[self.club_head_id]
        coriolis_force = -club_head_mass * coriolis_accel

        joint_coriolis = self.compute_coriolis_forces(qpos, qvel)
        apparent_force = jacp_curr.T @ joint_coriolis[: self.model.nv]

        centrifugal_direction = club_pos / (
            np.linalg.norm(club_pos) + EPSILON_SINGULARITY_DETECTION
        )
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
        if qpos is None:
            raise ValueError("qpos must be provided")
        coriolis_forces = self.compute_coriolis_forces(qpos, qvel)
        coriolis_power = np.dot(coriolis_forces, qvel)

        centrifugal, coupling = self.decompose_coriolis_forces(qpos, qvel)
        centrifugal_power = np.dot(centrifugal, qvel)
        coupling_power = np.dot(coupling, qvel)

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
        if qpos is None:
            raise ValueError("qpos must be provided")
        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)

        rotational_ke = 0.0
        translational_ke = 0.0

        for i in range(1, self.model.nbody):
            body_mass = self.model.body_mass[i]
            body_inertia = self.model.body_inertia[i]

            jacp, jacr = self._compute_jacobian(i, data=self._perturb_data)

            v_linear = jacp @ qvel
            omega = jacr @ qvel

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

        Args:
            times: Time array [N]
            positions: Joint positions [N x nv]
            velocities: Joint velocities [N x nv]
            accelerations: Joint accelerations [N x nv]

        Returns:
            List of KinematicForceData for each time step
        """
        if times is None:
            raise ValueError("times must be provided")
        results = []

        for i in range(len(times)):
            qpos = positions[i]
            qvel = velocities[i]
            qacc = accelerations[i]

            coriolis = self.compute_coriolis_forces(qpos, qvel)
            gravity = self.compute_gravity_forces(qpos)
            centrifugal, coupling = self.decompose_coriolis_forces(qpos, qvel)

            club_coriolis, club_centrifugal, club_apparent = (
                self.compute_club_head_apparent_forces(qpos, qvel, qacc)
            )

            power_dict = self.compute_kinematic_power(qpos, qvel)
            ke_dict = self.compute_kinetic_energy_components(qpos, qvel)

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

    def _validate_effective_mass_direction(self, direction: np.ndarray) -> np.ndarray:
        return validate_effective_mass_direction(direction)

    def _check_mass_matrix_conditioning(self, M: np.ndarray) -> None:
        check_mass_matrix_conditioning(M)

    def _check_jacobian_rank(self, jacp: np.ndarray) -> None:
        check_jacobian_rank(jacp)

    def _compute_effective_mass_value(
        self, direction: np.ndarray, jacp: np.ndarray, M: np.ndarray
    ) -> float:
        return compute_effective_mass_value(direction, jacp, M)

    def compute_effective_mass(
        self,
        qpos: np.ndarray,
        direction: np.ndarray,
        body_id: int | None = None,
    ) -> float:
        """Compute effective mass in a given direction.

        Args:
            qpos: Joint positions [nv] (rad for revolute, m for prismatic)
            direction: Direction vector [3] (will be normalized)
            body_id: Body to compute for (default: club head)

        Returns:
            Effective mass in that direction [kg]
        """
        if qpos is None:
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
