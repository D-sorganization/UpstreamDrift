from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.core.physics_constants import GRAVITY_M_S2
from src.shared.python.pendulum_simulator.triple_pendulum_spec import (
    TRIPLE_PENDULUM_SPEC,
    TriplePendulumSegmentSpec,
)

FIXTURE_TIMESTEP_S = 0.001
TORQUE_LIMIT_NM = 25.0
_DERIV_EPS = 1e-6


@dataclass(frozen=True)
class DoublePendulumSpec:
    segments: tuple[TriplePendulumSegmentSpec, TriplePendulumSegmentSpec]
    damping_nm_s_rad: tuple[float, float]
    gravity_m_s2: float = float(GRAVITY_M_S2)
    gravity_plane: str = "xz"


_SEG0, _SEG1, _ = TRIPLE_PENDULUM_SPEC.segments
OPTION_A_DOUBLE_PENDULUM_SPEC = DoublePendulumSpec(
    (_SEG0, _SEG1),
    (
        TRIPLE_PENDULUM_SPEC.damping_nm_s_rad[0],
        TRIPLE_PENDULUM_SPEC.damping_nm_s_rad[1],
    ),
    gravity_m_s2=TRIPLE_PENDULUM_SPEC.gravity_m_s2,
)


def build_mujoco_model(
    spec: DoublePendulumSpec = OPTION_A_DOUBLE_PENDULUM_SPEC,
    *,
    timestep_s: float = FIXTURE_TIMESTEP_S,
) -> Any:
    import mujoco as mj

    if timestep_s <= 0:
        raise ValueError("timestep_s must be positive")
    model = mj.MjModel.from_xml_string(
        double_pendulum_mjcf(spec, timestep_s=timestep_s)
    )
    if model.opt.integrator != mj.mjtIntegrator.mjINT_EULER or model.nu != 2:
        raise ValueError("fixture requires Euler 2-DOF plant")
    return model


def double_pendulum_mjcf(
    spec: DoublePendulumSpec,
    *,
    timestep_s: float,
    torque_limit_nm: float = TORQUE_LIMIT_NM,
) -> str:
    axis = "0 1 0"
    (s0, s1) = spec.segments
    d0, d1 = spec.damping_nm_s_rad
    l1, m1, c1, i1 = (
        s0.length_m,
        s0.mass_kg,
        s0.center_of_mass_ratio,
        s0.inertia_about_com,
    )
    l2, m2, c2, i2 = (
        s1.length_m,
        s1.mass_kg,
        s1.center_of_mass_ratio,
        s1.inertia_about_com,
    )
    lim = float(torque_limit_nm)
    g = spec.gravity_m_s2
    return rf"""<mujoco model="option_a_double_pendulum">
  <option timestep="{timestep_s}" gravity="0 0 -{g}" integrator="Euler"/>
  <worldbody>
    <body name="proximal" pos="0 0 1.0">
      <joint name="j1" type="hinge" axis="{axis}" damping="{d0}"/>
      <inertial pos="0 0 {-l1 * c1}" mass="{m1}" diaginertia="{i1 * 0.5} {i1} {i1}"/>
      <geom type="capsule" fromto="0 0 0 0 0 {-l1}" size="0.03" contype="0" conaffinity="0"/>
      <body name="distal" pos="0 0 {-l1}">
        <joint name="j2" type="hinge" axis="{axis}" damping="{d1}"/>
        <inertial pos="0 0 {-l2 * c2}" mass="{m2}" diaginertia="{i2 * 0.5} {i2} {i2}"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-l2}" size="0.02" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="j1" gear="1" ctrllimited="true" ctrlrange="{-lim} {lim}"/>
    <motor name="m2" joint="j2" gear="1" ctrllimited="true" ctrlrange="{-lim} {lim}"/>
  </actuator>
</mujoco>"""


def linearize_step(
    model: Any, data: Any, ctrl: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco as mj

    dim = 2 * model.nv
    a = np.empty((dim, dim), dtype=np.float64)
    b = np.empty((dim, model.nu), dtype=np.float64)
    data.ctrl[:] = ctrl
    mj.mj_forward(model, data)
    mj.mjd_transitionFD(model, data, _DERIV_EPS, True, a, b, None, None)
    return a, b


def transition_fd_agrees_with_rollout(
    model: Any,
    states: list[tuple[np.ndarray, np.ndarray]],
    *,
    ctrl: np.ndarray | None = None,
    atol: float = 5e-3,
) -> None:
    import mujoco as mj

    command = (
        np.asarray(ctrl, dtype=np.float64)
        if ctrl is not None
        else np.array([1.0, -0.5], dtype=np.float64)
    )
    if command.shape != (model.nu,) or np.any(
        np.abs(command) >= TORQUE_LIMIT_NM - _DERIV_EPS
    ):
        raise ValueError("ctrl must be interior to actuator bounds")

    def _step(ctrl: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        data = mj.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        mj.mj_step(model, data)
        return np.concatenate([data.qpos, data.qvel])

    for qpos, qvel in states:
        data = mj.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        a, b = linearize_step(model, data, command)
        x = np.concatenate([qpos, qvel])
        for j in range(model.nu):
            du = np.zeros(model.nu)
            du[j] = _DERIV_EPS
            fd = (_step(command + du, qpos, qvel) - _step(command - du, qpos, qvel)) / (
                2.0 * _DERIV_EPS
            )
            if not np.allclose(b[:, j], fd, atol=atol, rtol=0.0):
                raise ValueError("B column disagrees with finite-difference rollout")
        if not np.allclose(
            a @ x + b @ command, _step(command, qpos, qvel), atol=atol, rtol=0.0
        ):
            raise ValueError("A@x + B@u disagrees with finite-difference rollout")


def gravity_bias_at_pose(model: Any, qpos: np.ndarray) -> np.ndarray:
    import mujoco as mj

    data = mj.MjData(model)
    data.qpos[:] = qpos
    mj.mj_forward(model, data)
    return np.array(data.qfrc_bias, dtype=np.float64)


def distal_tip_position_m(
    qpos: np.ndarray,
    spec: DoublePendulumSpec = OPTION_A_DOUBLE_PENDULUM_SPEC,
    *,
    anchor_z_m: float = 1.0,
) -> np.ndarray:
    """Planar task-space position (x, z) of the distal link tip in metres."""
    if qpos.shape != (2,):
        raise ValueError("qpos must be length-2 for the Option A plant")
    q0, q1 = float(qpos[0]), float(qpos[1])
    l1, l2 = spec.segments[0].length_m, spec.segments[1].length_m
    end1_x = l1 * np.sin(q0)
    end1_z = anchor_z_m - l1 * np.cos(q0)
    q_sum = q0 + q1
    tip_x = end1_x + l2 * np.sin(q_sum)
    tip_z = end1_z - l2 * np.cos(q_sum)
    return np.array([tip_x, tip_z], dtype=np.float64)
