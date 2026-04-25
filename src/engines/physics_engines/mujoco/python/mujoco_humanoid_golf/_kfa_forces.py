from __future__ import annotations

import mujoco
import numpy as np

from src.shared.python.core.numerical_constants import (
    EPSILON_FINITE_DIFF_JACOBIAN,
)


class _KFAForcesMixin:
    model: mujoco.MjModel
    _perturb_data: mujoco.MjData

    def compute_coriolis_forces(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        return self.compute_coriolis_forces_rne(qpos, qvel)

    def compute_coriolis_forces_rne(
        self, qpos: np.ndarray, qvel: np.ndarray
    ) -> np.ndarray:
        if not (qpos is not None):
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
        if not (qpos is not None):
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
        if not (qpos is not None):
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
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        self._perturb_data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self._perturb_data)

        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self._perturb_data.qM)

        return M

    def compute_coriolis_matrix(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        if not (qpos is not None):
            raise ValueError("qpos must be provided")
        epsilon = EPSILON_FINITE_DIFF_JACOBIAN
        C = np.zeros((self.model.nv, self.model.nv))

        c_ref = self.compute_coriolis_forces(qpos, qvel)

        for i in range(self.model.nv):
            qvel_perturb = qvel.copy()
            qvel_perturb[i] += epsilon

            c_perturb = self.compute_coriolis_forces(qpos, qvel_perturb)

            C[:, i] = (c_perturb - c_ref) / epsilon

        return C
