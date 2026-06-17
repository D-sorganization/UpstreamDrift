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
        self._coriolis_cache_key: tuple[bytes, bytes] | None = None
        self._coriolis_cache_value: np.ndarray | None = None

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

    def _validate_vector(
        self,
        name: str,
        values: np.ndarray,
        expected_size: int,
    ) -> np.ndarray:
        """Validate a finite one-dimensional vector at a public boundary."""
        if values is None:
            raise ValueError(f"{name} must be provided")

        array = np.asarray(values, dtype=float)
        expected_shape = (expected_size,)
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    def _validate_qpos(self, qpos: np.ndarray) -> np.ndarray:
        return self._validate_vector("qpos", qpos, self.model.nq)

    def _validate_qvel(self, qvel: np.ndarray) -> np.ndarray:
        return self._validate_vector("qvel", qvel, self.model.nv)

    def _validate_qacc(self, qacc: np.ndarray) -> np.ndarray:
        return self._validate_vector("qacc", qacc, self.model.nv)

    def _validate_force_vector(self, name: str, values: np.ndarray) -> np.ndarray:
        return self._validate_vector(name, values, self.model.nv)

    def _validate_force_output(self, name: str, values: np.ndarray) -> np.ndarray:
        array = self._validate_force_vector(name, values)
        return array.copy()

    def _coriolis_cache_matches(self, qpos: np.ndarray, qvel: np.ndarray) -> bool:
        return self._coriolis_cache_key == (qpos.tobytes(), qvel.tobytes())

    def _read_cached_coriolis(
        self, qpos: np.ndarray, qvel: np.ndarray
    ) -> np.ndarray | None:
        if (
            self._coriolis_cache_matches(qpos, qvel)
            and self._coriolis_cache_value is not None
        ):
            return self._coriolis_cache_value.copy()
        return None

    def _write_cached_coriolis(
        self, qpos: np.ndarray, qvel: np.ndarray, coriolis: np.ndarray
    ) -> np.ndarray:
        value = self._validate_force_output("coriolis forces", coriolis)
        self._coriolis_cache_key = (qpos.tobytes(), qvel.tobytes())
        self._coriolis_cache_value = value.copy()
        return value.copy()

    def _validate_trajectory_inputs(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if times is None:
            raise ValueError("times must be provided")

        times_array = np.asarray(times, dtype=float)
        positions_array = np.asarray(positions, dtype=float)
        velocities_array = np.asarray(velocities, dtype=float)
        accelerations_array = np.asarray(accelerations, dtype=float)

        if times_array.ndim != 1:
            raise ValueError(f"times must have shape (N,), got {times_array.shape}")
        n_steps = times_array.shape[0]
        expected_positions = (n_steps, self.model.nq)
        expected_velocities = (n_steps, self.model.nv)
        if positions_array.shape != expected_positions:
            raise ValueError(
                f"positions must have shape {expected_positions}, "
                f"got {positions_array.shape}"
            )
        if velocities_array.shape != expected_velocities:
            raise ValueError(
                f"velocities must have shape {expected_velocities}, "
                f"got {velocities_array.shape}"
            )
        if accelerations_array.shape != expected_velocities:
            raise ValueError(
                f"accelerations must have shape {expected_velocities}, "
                f"got {accelerations_array.shape}"
            )

        for name, array in (
            ("times", times_array),
            ("positions", positions_array),
            ("velocities", velocities_array),
            ("accelerations", accelerations_array),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")

        return times_array, positions_array, velocities_array, accelerations_array

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

    def _clamp_to_joint_limits(self, qpos: np.ndarray) -> np.ndarray:
        """Clamp joint positions to the model's limited joint ranges."""
        qpos = self._validate_qpos(qpos)

        q_clamped = qpos.copy()
        for joint_id in range(self.model.njnt):
            if not self.model.jnt_limited[joint_id]:
                continue

            qpos_addr = self.model.jnt_qposadr[joint_id]
            if qpos_addr >= len(q_clamped):
                continue

            q_min = self.model.jnt_range[joint_id, 0]
            q_max = self.model.jnt_range[joint_id, 1]
            q_clamped[qpos_addr] = np.clip(q_clamped[qpos_addr], q_min, q_max)

        return q_clamped

    def compute_coriolis_forces(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Compute Coriolis and centrifugal forces.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Coriolis forces [nv]
        """
        qpos = self._validate_qpos(qpos)
        qvel = self._validate_qvel(qvel)

        cached = self._read_cached_coriolis(qpos, qvel)
        if cached is not None:
            return cached

        coriolis = self.compute_coriolis_forces_rne(qpos, qvel)
        return self._write_cached_coriolis(qpos, qvel, coriolis)

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
        qpos = self._validate_qpos(qpos)
        qvel = self._validate_qvel(qvel)

        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel
        self._perturb_data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self._perturb_data)

        bias = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, self._perturb_data, 0, bias)

        self._perturb_data.qvel[:] = 0.0
        self._perturb_data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self._perturb_data)
        gravity = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, self._perturb_data, 0, gravity)

        return self._validate_force_output("coriolis forces", bias - gravity)

    def compute_gravity_forces(self, qpos: np.ndarray) -> np.ndarray:
        """Compute gravitational forces.

        Args:
            qpos: Joint positions [nv]

        Returns:
            Gravity forces [nv]
        """
        qpos = self._validate_qpos(qpos)
        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self._perturb_data)
        return self._validate_force_output(
            "gravity forces", np.asarray(self._perturb_data.qfrc_bias.copy())
        )

    def decompose_coriolis_forces(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        coriolis_forces: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompose Coriolis forces into centrifugal and velocity coupling.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            coriolis_forces: Optional precomputed Coriolis forces [nv]

        Returns:
            Tuple of (centrifugal_forces [nv], coupling_forces [nv])
        """
        qpos = self._validate_qpos(qpos)
        qvel = self._validate_qvel(qvel)
        centrifugal = np.zeros(self.model.nv)
        if coriolis_forces is None:
            total_coriolis = self.compute_coriolis_forces(qpos, qvel)
        else:
            total_coriolis = self._validate_force_vector(
                "coriolis_forces", coriolis_forces
            )

        for i in range(self.model.nv):
            qvel_single = np.zeros(self.model.nv)
            qvel_single[i] = qvel[i]
            single_coriolis = self.compute_coriolis_forces(qpos, qvel_single)
            centrifugal += single_coriolis

        coupling = total_coriolis - centrifugal
        return (
            self._validate_force_output("centrifugal forces", centrifugal),
            self._validate_force_output("velocity coupling forces", coupling),
        )

    def compute_mass_matrix(self, qpos: np.ndarray) -> np.ndarray:
        """Compute configuration-dependent mass matrix M(q).

        Args:
            qpos: Joint positions [nv]

        Returns:
            Mass matrix [nv x nv]
        """
        qpos = self._validate_qpos(qpos)
        return compute_mass_matrix(self.model, self._perturb_data, qpos)

    def compute_coriolis_matrix(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Compute Coriolis matrix C(q,q̇).

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Coriolis matrix [nv x nv]
        """
        qpos = self._validate_qpos(qpos)
        qvel = self._validate_qvel(qvel)
        return compute_coriolis_matrix(
            self.model, qpos, qvel, self.compute_coriolis_forces
        )

    def compute_club_head_apparent_forces(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        *,
        coriolis_forces: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute apparent forces at club head.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            qacc: Joint accelerations [nv]

        Returns:
            Tuple of (coriolis_force [3], centrifugal_force [3], total_apparent [3])
        """
        qpos = self._validate_qpos(qpos)
        qvel = self._validate_qvel(qvel)
        self._validate_qacc(qacc)
        if self.club_head_id is None:
            return np.zeros(3), np.zeros(3), np.zeros(3)

        qpos = self._clamp_to_joint_limits(qpos)

        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)

        jacp, _ = self._compute_jacobian(self.club_head_id, data=self._perturb_data)
        jacp_curr = jacp.copy()

        club_pos = self._perturb_data.xpos[self.club_head_id].copy()
        epsilon = EPSILON_FINITE_DIFF_JACOBIAN

        qpos_forward = self._clamp_to_joint_limits(qpos + epsilon * qvel)
        self._perturb_data.qpos[:] = qpos_forward
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)
        jacp_forward, _ = self._compute_jacobian(
            self.club_head_id, data=self._perturb_data
        )
        jacp_forward = jacp_forward.copy()

        qpos_backward = self._clamp_to_joint_limits(qpos - epsilon * qvel)
        self._perturb_data.qpos[:] = qpos_backward
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)
        jacp_backward, _ = self._compute_jacobian(
            self.club_head_id, data=self._perturb_data
        )

        # Use the effective post-clamp step along the qvel direction rather than
        # the nominal 2 * epsilon denominator. Clamping can make one-sided
        # (or zero-sided) perturbations at joint limits; dividing by the fixed
        # nominal step then systematically underestimates jacp_dot. Projecting
        # the actual displacement (qpos_forward - qpos_backward) onto qvel and
        # dividing by ||qvel||^2 recovers the scalar step along qvel. Falls back
        # to the nominal denominator when qvel is (near-)zero or the effective
        # step collapses to zero (both sides clamped to the same limit).
        qvel_norm_sq = float(np.dot(qvel, qvel))
        if qvel_norm_sq > EPSILON_SINGULARITY_DETECTION:
            effective_step = float(
                np.dot(qpos_forward - qpos_backward, qvel) / qvel_norm_sq
            )
            if abs(effective_step) < EPSILON_SINGULARITY_DETECTION:
                effective_step = 2.0 * epsilon
        else:
            effective_step = 2.0 * epsilon

        jacp_dot = (jacp_forward - jacp_backward) / effective_step
        coriolis_accel = jacp_dot @ qvel

        club_head_mass = self.model.body_mass[self.club_head_id]
        coriolis_force = -club_head_mass * coriolis_accel

        if coriolis_forces is None:
            joint_coriolis = self.compute_coriolis_forces(qpos, qvel)
        else:
            joint_coriolis = self._validate_force_vector(
                "coriolis_forces", coriolis_forces
            )
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
        *,
        coriolis_forces: np.ndarray | None = None,
        decomposition: tuple[np.ndarray, np.ndarray] | None = None,
        gravity_forces: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute power contributions from kinematic forces.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Dictionary with power contributions
        """
        qpos = self._validate_qpos(qpos)
        qvel = self._validate_qvel(qvel)
        if coriolis_forces is None:
            coriolis = self.compute_coriolis_forces(qpos, qvel)
        else:
            coriolis = self._validate_force_vector("coriolis_forces", coriolis_forces)
        coriolis_power = np.dot(coriolis, qvel)

        if decomposition is None:
            centrifugal, coupling = self.decompose_coriolis_forces(
                qpos, qvel, coriolis_forces=coriolis
            )
        else:
            centrifugal = self._validate_force_vector(
                "centrifugal forces", decomposition[0]
            )
            coupling = self._validate_force_vector(
                "velocity coupling forces", decomposition[1]
            )
        centrifugal_power = np.dot(centrifugal, qvel)
        coupling_power = np.dot(coupling, qvel)

        if gravity_forces is None:
            gravity = self.compute_gravity_forces(qpos)
        else:
            gravity = self._validate_force_vector("gravity_forces", gravity_forces)
        gravity_power = np.dot(gravity, qvel)

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
        qpos = self._validate_qpos(qpos)
        qvel = self._validate_qvel(qvel)
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
        times, positions, velocities, accelerations = self._validate_trajectory_inputs(
            times, positions, velocities, accelerations
        )
        results = []

        for i in range(len(times)):
            qpos = positions[i]
            qvel = velocities[i]
            qacc = accelerations[i]

            coriolis = self.compute_coriolis_forces(qpos, qvel)
            gravity = self.compute_gravity_forces(qpos)
            centrifugal, coupling = self.decompose_coriolis_forces(
                qpos, qvel, coriolis_forces=coriolis
            )

            club_coriolis, club_centrifugal, club_apparent = (
                self.compute_club_head_apparent_forces(
                    qpos, qvel, qacc, coriolis_forces=coriolis
                )
            )

            power_dict = self.compute_kinematic_power(
                qpos,
                qvel,
                coriolis_forces=coriolis,
                decomposition=(centrifugal, coupling),
                gravity_forces=gravity,
            )
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
        qpos = self._validate_qpos(qpos)
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
