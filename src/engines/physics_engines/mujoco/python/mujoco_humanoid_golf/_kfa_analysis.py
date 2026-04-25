from __future__ import annotations

import mujoco
import numpy as np

from src.shared.python.core.numerical_constants import (
    EPSILON_FINITE_DIFF_JACOBIAN,
    EPSILON_SINGULARITY_DETECTION,
)

from ._kinematic_force_data import KinematicForceData


class _KFAAnalysisMixin:
    model: mujoco.MjModel
    _perturb_data: mujoco.MjData
    club_head_id: int | None

    def compute_club_head_apparent_forces(  # noqa: PLR0915
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        if self.club_head_id is None:
            return np.zeros(3), np.zeros(3), np.zeros(3)

        # Clamp the reference state to valid joint limits before any evaluation.
        qpos = self._clamp_to_joint_limits(qpos)

        self._perturb_data.qpos[:] = qpos
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)

        jacp, _ = self._compute_jacobian(self.club_head_id, data=self._perturb_data)
        jacp_curr = jacp.copy()

        club_pos = self._perturb_data.xpos[self.club_head_id].copy()

        epsilon = EPSILON_FINITE_DIFF_JACOBIAN

        # Clamp the perturbed states and measure the *effective* step so that
        # the finite-difference denominator matches the actual perturbation when
        # a joint is at its limit and one side clips.
        qpos_fwd = self._clamp_to_joint_limits(qpos + epsilon * qvel)
        qpos_bwd = self._clamp_to_joint_limits(qpos - epsilon * qvel)
        # Per-DOF effective half-step; fall back to epsilon where qvel is zero.
        step_fwd = qpos_fwd - qpos
        step_bwd = qpos - qpos_bwd
        # Total step size across each DOF; avoid division by zero.
        total_step = step_fwd + step_bwd
        effective_denom = np.where(np.abs(total_step) > 0.0, total_step, 2.0 * epsilon)

        self._perturb_data.qpos[:] = qpos_fwd
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)

        jacp_forward, _ = self._compute_jacobian(
            self.club_head_id, data=self._perturb_data
        )
        jacp_forward = jacp_forward.copy()

        self._perturb_data.qpos[:] = qpos_bwd
        self._perturb_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self._perturb_data)

        jacp_backward, _ = self._compute_jacobian(
            self.club_head_id, data=self._perturb_data
        )

        # Broadcast effective_denom (shape nv) across the 3-row Jacobian rows.
        jacp_dot = (jacp_forward - jacp_backward) / effective_denom[np.newaxis, :]

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
        if not (qpos is not None):
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
        if not (qpos is not None):
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
        if not (times is not None):
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
