from __future__ import annotations

import mujoco
import numpy as np


class _KFACoreMixin:
    model: mujoco.MjModel
    data: mujoco.MjData
    _perturb_data: mujoco.MjData
    nv: int
    _use_reshaped_arrays: bool
    _jacp: np.ndarray
    _jacr: np.ndarray
    club_head_id: int | None
    club_grip_id: int | None

    def _init_core(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not (model is not None):
            raise ValueError("model must be provided")
        self.model = model
        self.data = data

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
        if not (name_pattern is not None):
            raise ValueError("name_pattern must be provided")
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and name_pattern.lower() in body_name.lower():
                return i
        return None

    def _compute_jacobian(
        self, body_id: int, data: mujoco.MjData | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not (body_id is not None):
            raise ValueError("body_id must be provided")
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
