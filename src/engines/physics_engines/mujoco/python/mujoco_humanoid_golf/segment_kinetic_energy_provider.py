"""MuJoCo adapter for the segment kinetic-energy overlay contract (#15389)."""

from __future__ import annotations

import mujoco
import numpy as np

from src.shared.python.segment_kinetic_energy.contracts import (
    SegmentKineticEnergyFrame,
    SegmentKineticEnergySample,
)

from .power_flow import PowerFlowAnalyzer

__all__ = ["MujocoSegmentKineticEnergyProvider"]


def _vec3(seq: np.ndarray) -> tuple[float, float, float]:
    return (float(seq[0]), float(seq[1]), float(seq[2]))


class MujocoSegmentKineticEnergyProvider:
    """Sample per-body translational, rotational and total KE from MuJoCo state."""

    ENGINE = "mujoco"

    def __init__(self, model: mujoco.MjModel) -> None:
        if not isinstance(model, mujoco.MjModel):
            raise TypeError("model must be mujoco.MjModel")
        self._model = model
        self._analyzer = PowerFlowAnalyzer(model)

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    def frame_at(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        time_s: float = 0.0,
    ) -> SegmentKineticEnergyFrame:
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        scratch = self._analyzer._data
        scratch.qpos[:] = qpos
        scratch.qvel[:] = qvel
        mujoco.mj_forward(self._model, scratch)
        mujoco.mj_energyVel(self._model, scratch)
        system_ke = float(scratch.energy[1])

        _total, ke_trans, ke_rot, _pe = self._analyzer._compute_segment_energies(qvel)

        segments: list[SegmentKineticEnergySample] = []
        for i in range(self._model.nbody):
            mass = float(self._model.body(i).mass[0])
            if mass <= 0.0:
                continue
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i) or (
                f"body_{i}"
            )
            trans_j = float(ke_trans[i])
            rot_j = float(ke_rot[i])
            total_j = trans_j + rot_j
            if total_j <= 0.0:
                continue
            segments.append(
                SegmentKineticEnergySample(
                    body=name,
                    body_index=i,
                    com_m=_vec3(scratch.xipos[i]),
                    translational_j=trans_j,
                    rotational_j=rot_j,
                    total_j=total_j,
                )
            )

        return SegmentKineticEnergyFrame(
            schema_version=1,
            time_s=float(time_s),
            engine=self.ENGINE,
            segments=tuple(segments),
            system_kinetic_j=system_ke,
        )
