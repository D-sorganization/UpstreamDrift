"""Admit and linearize the contact-free native RK4 hinge for BoxFDDP (F05b)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.native_torque_replay import (
    _validate_unit_motors,
)

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class HingeDiscreteModel:
    """Exact RK4 state and input Jacobians for the admitted linear native hinge."""

    A: Array
    B: Array
    inertia_kg_m2: float
    damping_nm_s_rad: float
    time_step_s: float

    def __post_init__(self) -> None:
        for name, shape in (("A", (2, 2)), ("B", (2, 1))):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"native hinge {name} must be finite {shape}")
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def _rk4_transition(inertia: float, damping: float, dt: float) -> tuple[Array, Array]:
    generator = np.array(
        [[0.0, 1.0, 0.0], [0.0, -damping / inertia, 1.0 / inertia], [0, 0, 0]]
    )
    scaled = dt * generator
    transition = np.eye(3)
    power = np.eye(3)
    for degree in range(1, 5):
        power = power @ scaled / degree
        transition += power
    return transition[:2, :2], transition[:2, 2:3]


def _native_step(model: Any, state: Array, torque: float) -> Array:
    import mujoco as mj

    data = mj.MjData(model)
    data.qpos[0], data.qvel[0] = state
    data.ctrl[0] = torque
    mj.mj_step(model, data)
    if (
        data.ncon
        or not np.isfinite(data.qpos).all()
        or not np.isfinite(data.qvel).all()
    ):
        raise ValueError("native hinge developed contact or a nonfinite state")
    return np.array([data.qpos[0], data.qvel[0]], dtype=np.float64)


def linearize_native_hinge(model: Any) -> HingeDiscreteModel:
    """Bind a strict direct-motor fixture to its analytic discrete Jacobians.

    MuJoCo's ``mjd_transitionFD`` does not support RK4. This restricted
    one-hinge/no-gravity/no-contact plant has linear continuous acceleration,
    so the RK4 polynomial is its exact discrete derivative. Two independent
    native steps guard implementation/model drift before solver admission.
    """
    import mujoco as mj

    if model.opt.integrator != mj.mjtIntegrator.mjINT_RK4:
        raise ValueError("native BoxFDDP hinge requires RK4 integration")
    if (
        model.nbody != 2
        or model.ngeom != 1
        or model.njnt != 1
        or model.nq != 1
        or model.nv != 1
        or model.nu != 1
        or model.na
        or model.nplugin
        or model.nmocap
        or model.neq
        or np.any(model.opt.gravity)
        or model.opt.disableactuator
        or np.any(model.body_gravcomp)
        or np.any(model.jnt_actfrclimited)
        or model.opt.disableflags & ~int(mj.mjtDisableBit.mjDSBL_AUTORESET)
    ):
        raise ValueError("native BoxFDDP hinge requires contact-free one-joint physics")
    _validate_unit_motors(model, mj)
    if (
        model.jnt_stiffness[0] != 0
        or model.dof_frictionloss[0] != 0
        or not model.actuator_ctrllimited[0]
        or model.actuator_ctrlrange[0, 0] >= model.actuator_ctrlrange[0, 1]
    ):
        raise ValueError("native hinge has unsupported passive or actuator force")
    data = mj.MjData(model)
    data.qpos[0], data.qvel[0] = 0.37, 0.91
    mj.mj_forward(model, data)
    inertia = float(data.qM[0])
    damping = float(model.dof_damping[0])
    dt = float(model.opt.timestep)
    if (
        not np.isfinite([inertia, damping, dt]).all()
        or inertia <= 0
        or damping < 0
        or dt <= 0
        or not np.isclose(data.qfrc_passive[0], -damping * data.qvel[0], atol=1e-11)
        or not np.isclose(data.qfrc_bias[0], 0.0, atol=1e-11)
    ):
        raise ValueError("native hinge inertia or passive dynamics are unsupported")
    A, B = _rk4_transition(inertia, damping, dt)
    for state, torque in ((np.array([0.4, -0.8]), 1.1), (np.array([-0.6, 0.5]), -0.7)):
        if not np.allclose(
            A @ state + B[:, 0] * torque,
            _native_step(model, state, torque),
            atol=1e-11,
            rtol=0,
        ):
            raise ValueError(
                "native RK4 step differs from admitted analytic derivative"
            )
    return HingeDiscreteModel(A, B, inertia, damping, dt)
