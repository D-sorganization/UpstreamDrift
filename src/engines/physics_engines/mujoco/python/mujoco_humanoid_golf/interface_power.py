"""MuJoCo Robertson–Winter interface power adapter (#15390)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.biomechanics.robertson_winter_interface_power import (
    InterfacePowerFrame,
    InterfacePowerJointSample,
    InterfacePowerSegment,
    segment_energy_balance_residual_w,
)

from .mujoco_rw_joint import joint_interface_powers_w

if TYPE_CHECKING:
    import mujoco

__all__ = ["MujocoInterfacePowerAnalyzer"]

_FD_STEP_S = 1e-4


class MujocoInterfacePowerAnalyzer:
    """Per-joint cfrc_int interface power with per-body energy balance."""

    ENGINE = "mujoco"

    def __init__(self, model: mujoco.MjModel) -> None:
        import mujoco

        if not isinstance(model, mujoco.MjModel):
            raise TypeError("model must be mujoco.MjModel")
        self._model = model
        self._scratch = mujoco.MjData(model)

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    def _prepare(self, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray):
        import mujoco

        model, scratch = self._model, self._scratch
        scratch.qpos[:] = np.asarray(qpos, dtype=np.float64)
        scratch.qvel[:] = np.asarray(qvel, dtype=np.float64)
        scratch.qacc[:] = np.asarray(qacc, dtype=np.float64)
        mujoco.mj_forward(model, scratch)
        mujoco.mj_inverse(model, scratch)
        scratch.qacc[:] = np.asarray(qacc, dtype=np.float64)
        mujoco.mj_rnePostConstraint(model, scratch)
        return model, scratch

    def _body_mechanical_energy(
        self,
        scratch: mujoco.MjData,
        body_id: int,
        qvel: np.ndarray,
    ) -> float:
        import mujoco

        model = self._model
        mass = float(model.body(body_id).mass[0])
        if mass <= 0.0:
            return 0.0
        gravity = np.asarray(model.opt.gravity, dtype=np.float64)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, scratch, jacp, jacr, body_id)
        com_vel = jacp @ qvel
        ang_vel_world = jacr @ qvel
        rot = np.asarray(scratch.ximat[body_id], dtype=np.float64).reshape(3, 3)
        ang_vel_body = rot.T @ ang_vel_world
        inertia = np.asarray(model.body(body_id).inertia, dtype=np.float64)
        ke = 0.5 * mass * float(np.dot(com_vel, com_vel))
        ke += 0.5 * float(np.dot(inertia, ang_vel_body * ang_vel_body))
        com_pos = np.asarray(scratch.xipos[body_id], dtype=np.float64)
        pe = -mass * float(np.dot(gravity, com_pos))
        return ke + pe

    def _finite_difference_me_rate(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        body_id: int,
    ) -> float:
        import mujoco

        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        qacc = np.asarray(qacc, dtype=np.float64)
        model, scratch = self._prepare(qpos, qvel, qacc)
        dt = _FD_STEP_S
        e0 = self._body_mechanical_energy(scratch, body_id, qvel)
        qpos1 = qpos + dt * qvel + 0.5 * (dt**2) * qacc
        qvel1 = qvel + dt * qacc
        scratch.qpos[:] = qpos1
        scratch.qvel[:] = qvel1
        scratch.qacc[:] = qacc
        mujoco.mj_forward(model, scratch)
        e1 = self._body_mechanical_energy(scratch, body_id, qvel1)
        return float((e1 - e0) / dt)

    def _body_actuator_power(
        self, scratch: mujoco.MjData, body_id: int, qvel: np.ndarray
    ) -> float:
        model = self._model
        power = 0.0
        for dof in range(model.nv):
            if int(model.dof_jntid[dof]) < 0:
                continue
            j = int(model.dof_jntid[dof])
            if int(model.jnt_bodyid[j]) != body_id:
                continue
            power += float(scratch.qfrc_actuator[dof] * qvel[dof])
        return power

    def _body_external_power(
        self, scratch: mujoco.MjData, body_id: int, qvel: np.ndarray
    ) -> float:
        import mujoco

        model = self._model
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, scratch, jacp, jacr, body_id)
        com_vel = jacp @ qvel
        ang_vel = jacr @ qvel
        wrench = scratch.xfrc_applied[body_id]
        return float(np.dot(wrench[:3], com_vel) + np.dot(wrench[3:], ang_vel))

    def frame_at(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        *,
        time_s: float = 0.0,
    ) -> InterfacePowerFrame:
        import mujoco

        model, scratch = self._prepare(qpos, qvel, qacc)
        qvel_a = np.asarray(qvel, dtype=np.float64)

        joint_samples: list[InterfacePowerJointSample] = []
        iface_on_body: dict[int, float] = {}

        for j in range(model.njnt):
            child_id = int(model.jnt_bodyid[j])
            parent_id = int(model.body_parentid[child_id])
            if child_id <= 0:
                continue
            (
                p_child,
                p_parent,
                fp_child,
                mp_child,
                coord_p,
            ) = joint_interface_powers_w(model, scratch, j, qvel_a)
            j_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
            )
            c_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, child_id)
                or f"body_{child_id}"
            )
            p_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
                or f"body_{parent_id}"
            )
            joint_samples.append(
                InterfacePowerJointSample(
                    joint=j_name,
                    joint_index=j,
                    child_body=c_name,
                    child_body_index=child_id,
                    parent_body=p_name,
                    parent_body_index=parent_id,
                    power_on_child_w=p_child,
                    power_on_parent_w=p_parent,
                    force_power_on_child_w=fp_child,
                    moment_power_on_child_w=mp_child,
                    joint_coordinate_power_w=coord_p,
                )
            )
            iface_on_body[child_id] = iface_on_body.get(child_id, 0.0) + p_child
            if parent_id > 0:
                iface_on_body[parent_id] = iface_on_body.get(parent_id, 0.0) + p_parent

        segments: list[InterfacePowerSegment] = []
        for body_id in range(1, model.nbody):
            mass = float(model.body(body_id).mass[0])
            if mass <= 0.0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or (
                f"body_{body_id}"
            )
            total_iface = float(iface_on_body.get(body_id, 0.0))
            actuator = self._body_actuator_power(scratch, body_id, qvel_a)
            external = self._body_external_power(scratch, body_id, qvel_a)
            me_rate = self._finite_difference_me_rate(qpos, qvel, qacc, body_id)
            residual = segment_energy_balance_residual_w(
                mechanical_energy_rate_w=me_rate,
                interface_power_w=total_iface,
                external_power_w=external,
                actuator_power_w=actuator,
            )
            segments.append(
                InterfacePowerSegment(
                    body=name,
                    body_index=body_id,
                    mass_kg=mass,
                    total_interface_power_w=total_iface,
                    external_power_w=external,
                    actuator_power_w=actuator,
                    mechanical_energy_rate_w=me_rate,
                    balance_residual_w=residual,
                )
            )

        return InterfacePowerFrame(
            schema_version=1,
            time_s=float(time_s),
            engine=self.ENGINE,
            joints=tuple(joint_samples),
            segments=tuple(segments),
        )
