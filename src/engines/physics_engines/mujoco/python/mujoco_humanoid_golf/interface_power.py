"""MuJoCo Robertson–Winter interface power adapter (#15390)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.biomechanics.robertson_winter_interface_power import (
    InterfacePowerFrame,
    InterfacePowerSegment,
    segment_energy_balance_residual_w,
)

if TYPE_CHECKING:
    import mujoco

__all__ = ["MujocoInterfacePowerAnalyzer"]


def _vec3(seq: np.ndarray) -> tuple[float, float, float]:
    return (float(seq[0]), float(seq[1]), float(seq[2]))


class MujocoInterfacePowerAnalyzer:
    """Compute RW interface power and per-segment mechanical-energy balance."""

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

    def _robertson_winter_reaction(
        self,
        scratch: mujoco.MjData,
        body_id: int,
        qvel: np.ndarray,
    ) -> tuple[float, float]:
        """Equivalent joint reaction (planar ``m a - m g``) and moment power at the joint."""
        model = self._model
        import mujoco

        mass = float(model.body(body_id).mass[0])
        gravity = np.asarray(model.opt.gravity, dtype=np.float64)
        lin_acc = np.asarray(scratch.cacc[body_id, :3], dtype=np.float64)
        ang_acc_world = np.asarray(scratch.cacc[body_id, 3:], dtype=np.float64)
        reaction_force = mass * lin_acc - mass * gravity

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, scratch, jacp, jacr, body_id)
        lin_vel = jacp @ qvel
        ang_vel = jacr @ qvel

        rot = np.asarray(scratch.ximat[body_id], dtype=np.float64).reshape(3, 3)
        omega_body = rot.T @ ang_vel
        alpha_body = rot.T @ ang_acc_world
        inertia = np.asarray(model.body(body_id).inertia, dtype=np.float64)
        net_moment = inertia * alpha_body + np.cross(omega_body, inertia * omega_body)
        return (
            float(np.dot(reaction_force, lin_vel)),
            float(np.dot(net_moment, omega_body)),
        )

    def _segment_mechanical_rate(
        self,
        scratch: mujoco.MjData,
        body_id: int,
        qvel: np.ndarray,
    ) -> tuple[float, float, float]:
        import mujoco

        model = self._model
        body = model.body(body_id)
        mass = float(body.mass[0])
        if mass <= 0.0:
            return 0.0, 0.0, 0.0
        gravity = np.asarray(model.opt.gravity, dtype=np.float64)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, scratch, jacp, jacr, body_id)
        com_vel = jacp @ qvel
        ang_vel_world = jacr @ qvel

        lin_acc = np.asarray(scratch.cacc[body_id, :3], dtype=np.float64)
        ang_acc_world = np.asarray(scratch.cacc[body_id, 3:], dtype=np.float64)
        rot = np.asarray(scratch.ximat[body_id], dtype=np.float64).reshape(3, 3)
        ang_vel_body = rot.T @ ang_vel_world
        ang_acc_body = rot.T @ ang_acc_world
        inertia = np.asarray(body.inertia, dtype=np.float64)
        ke_rate = mass * float(np.dot(com_vel, lin_acc)) + float(
            np.dot(inertia, ang_vel_body * ang_acc_body)
        )
        gravity_power = mass * float(np.dot(gravity, com_vel))
        external = float(np.dot(scratch.xfrc_applied[body_id, :3], com_vel)) + float(
            np.dot(scratch.xfrc_applied[body_id, 3:], ang_vel_world)
        )
        return ke_rate, gravity_power, external

    def frame_at(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        *,
        time_s: float = 0.0,
    ) -> InterfacePowerFrame:
        import mujoco

        model, scratch = self._model, self._scratch
        scratch.qpos[:] = np.asarray(qpos, dtype=np.float64)
        scratch.qvel[:] = np.asarray(qvel, dtype=np.float64)
        scratch.qacc[:] = np.asarray(qacc, dtype=np.float64)
        mujoco.mj_forward(model, scratch)
        mujoco.mj_inverse(model, scratch)
        scratch.qacc[:] = np.asarray(qacc, dtype=np.float64)
        mujoco.mj_rnePostConstraint(model, scratch)

        segments: list[InterfacePowerSegment] = []
        for body_id in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or (
                f"body_{body_id}"
            )
            mass = float(model.body(body_id).mass[0])
            if mass <= 0.0:
                continue
            force_power, moment_power = self._robertson_winter_reaction(
                scratch, body_id, np.asarray(qvel, dtype=np.float64)
            )
            total_interface = force_power + moment_power
            ke_rate, gravity_power, external_power = self._segment_mechanical_rate(
                scratch, body_id, np.asarray(qvel, dtype=np.float64)
            )
            me_rate = ke_rate - gravity_power
            residual = segment_energy_balance_residual_w(
                mechanical_energy_rate_w=ke_rate,
                interface_power_w=total_interface,
                gravity_power_w=gravity_power,
                external_power_w=external_power,
            )
            segments.append(
                InterfacePowerSegment(
                    body=name,
                    body_index=body_id,
                    mass_kg=mass,
                    force_power_w=force_power,
                    moment_power_w=moment_power,
                    total_interface_power_w=total_interface,
                    gravity_power_w=gravity_power,
                    external_power_w=external_power,
                    mechanical_energy_rate_w=me_rate,
                    balance_residual_w=residual,
                )
            )
        return InterfacePowerFrame(
            schema_version=1,
            time_s=float(time_s),
            engine=self.ENGINE,
            segments=tuple(segments),
        )
