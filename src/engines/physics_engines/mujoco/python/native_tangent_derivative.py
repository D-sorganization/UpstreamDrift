"""Strict native discrete tangent derivatives for a floating two-hinge fixture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.native_torque_replay import (
    _validate_unit_motors,
)

Array: TypeAlias = NDArray[np.float64]


def _frozen(values: Array) -> Array:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NativeTangentStep:
    """Native Euler Jacobians on ``[dq_tangent, dv]`` and stepped full state."""

    A: Array
    B: Array
    next_qpos: Array
    next_qvel: Array
    next_integration_state: Array
    time_step_s: float
    ordered_input_channel_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        nv = self.next_qvel.size
        nu = len(self.ordered_input_channel_ids)
        for name, shape in (
            ("A", (2 * nv, 2 * nv)),
            ("B", (2 * nv, nu)),
            ("next_qpos", (nv + 1,)),
            ("next_qvel", (nv,)),
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"native tangent {name} requires finite shape {shape}")
            object.__setattr__(self, name, _frozen(values))
        full = np.asarray(self.next_integration_state, dtype=np.float64)
        if full.ndim != 1 or not np.isfinite(full).all():
            raise ValueError("next native integration state must be finite and flat")
        object.__setattr__(self, "next_integration_state", _frozen(full))
        if not np.isfinite(self.time_step_s) or self.time_step_s <= 0:
            raise ValueError("native step must have a finite positive time")


def _admit_fixture(model: Any, command: Array, epsilon: float) -> tuple[str, ...]:
    import mujoco as mj

    if model.opt.integrator != mj.mjtIntegrator.mjINT_EULER:
        raise ValueError("native tangent derivative requires Euler integration")
    if (
        model.ntendon
        or model.nflex
        or np.any(model.geom_contype)
        or np.any(model.geom_conaffinity)
    ):
        raise ValueError("native tangent derivative requires contact-free geometry")
    if (
        model.nq != 9
        or model.nv != 8
        or model.nu != 2
        or model.na
        or model.njnt != 3
        or model.nplugin
        or model.nmocap
        or model.neq
        or np.any(model.opt.gravity)
        or np.any(model.body_gravcomp)
        or model.opt.disableactuator
        or np.any(model.jnt_actfrclimited)
        or model.opt.disableflags & ~int(mj.mjtDisableBit.mjDSBL_AUTORESET)
        or model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE
        or np.any(model.jnt_type[1:] != mj.mjtJoint.mjJNT_HINGE)
    ):
        raise ValueError(
            "native tangent derivative requires unforced floating two-hinge fixture"
        )
    _validate_unit_motors(model, mj)
    if set(model.actuator_trnid[:, 0]) != {1, 2} or not np.all(
        model.actuator_ctrllimited
    ):
        raise ValueError("two distinct bounded direct hinge motors are required")
    controls = np.asarray(command, dtype=np.float64)
    if controls.shape != (2,) or not np.isfinite(controls).all():
        raise ValueError("native torque command requires two finite channels")
    if np.any(controls < model.actuator_ctrlrange[:, 0]) or np.any(
        controls > model.actuator_ctrlrange[:, 1]
    ):
        raise ValueError("native torque exceeds post-limit actuator bounds")
    if np.any(controls <= model.actuator_ctrlrange[:, 0] + epsilon) or np.any(
        controls >= model.actuator_ctrlrange[:, 1] - epsilon
    ):
        raise ValueError("native central derivative requires interior actuator torque")
    return tuple(
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    )


def _restore(model: Any, full_state: Array, mj: Any) -> Any:
    state = np.asarray(full_state, dtype=np.float64)
    size = mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION)
    if state.shape != (size,) or not np.isfinite(state).all():
        raise ValueError("complete finite native integration state is required")
    data = mj.MjData(model)
    mj.mj_setState(model, data, state, mj.mjtState.mjSTATE_INTEGRATION)
    normalized = data.qpos.copy()
    mj.mj_normalizeQuat(model, normalized)
    if (
        not np.allclose(data.qpos, normalized, atol=1e-12, rtol=0)
        or data.time != 0
        or np.any(data.qfrc_applied)
        or np.any(data.xfrc_applied)
    ):
        raise ValueError("native initial state has quaternion, time or load mismatch")
    return data


def linearize_native_tangent_step(
    model: Any,
    initial_integration_state: Array,
    applied_motor_torques: Array,
    *,
    epsilon: float = 1e-6,
) -> NativeTangentStep:
    """Differentiate the actual native Euler step in its configuration tangent.

    This is MuJoCo's finite-differenced discrete derivative, not a continuous
    derivative integrated by a different numerical scheme. A fresh instance
    supplies the independently stepped output at the same held motor command.
    """
    import mujoco as mj

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("finite-difference epsilon must be positive and finite")
    channels = _admit_fixture(model, applied_motor_torques, epsilon)
    model.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_AUTORESET)
    data = _restore(model, initial_integration_state, mj)
    data.ctrl[:] = applied_motor_torques
    mj.mj_forward(model, data)
    if data.ncon or not np.isfinite(data.qacc).all():
        raise ValueError("contact or nonfinite acceleration blocks tangent derivative")
    dimension = 2 * model.nv
    A = np.empty((dimension, dimension), dtype=np.float64)
    B = np.empty((dimension, model.nu), dtype=np.float64)
    mj.mjd_transitionFD(model, data, epsilon, True, A, B, None, None)
    stepped = _restore(model, initial_integration_state, mj)
    stepped.ctrl[:] = applied_motor_torques
    mj.mj_step(model, stepped)
    if (
        stepped.ncon
        or not np.isfinite(stepped.qpos).all()
        or not np.isfinite(stepped.qvel).all()
        or not np.isclose(stepped.time, model.opt.timestep, atol=1e-12, rtol=0)
    ):
        raise ValueError("native tangent step contacted, reset or diverged")
    full_next = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, stepped, full_next, mj.mjtState.mjSTATE_INTEGRATION)
    return NativeTangentStep(
        A,
        B,
        stepped.qpos,
        stepped.qvel,
        full_next,
        float(model.opt.timestep),
        channels,
    )
