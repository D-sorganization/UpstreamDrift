"""Per-frame inverse dynamics and reaction extraction for lift joint reactions."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python.msk_lift_joint_reaction_diagnostics import (
    record_max_udot_error,
)
from src.engines.physics_engines.opensim.python.msk_lift_simbody_kinematics import (
    simbody_kinematics_at_spline_time,
)
from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
    transform_to_matrix,
)
from src.engines.physics_engines.opensim.python.spine.joint_reactions import (
    _child_reaction,
    _override,
)
from src.shared.python.engine_core.opensim_runtime import import_opensim

__all__ = [
    "collect_frame_reaction_loads",
    "solve_lift_frame_inverse_dynamics",
]


def solve_lift_frame_inverse_dynamics(
    model: Any,
    state: Any,
    context: dict[str, Any],
    row: int,
) -> tuple[np.ndarray, float, float]:
    """Apply kinematics, ID residuals, and return (udot, acc_err, udot_norm)."""
    osim = import_opensim()
    splines = context["splines"]
    ik = context["ik"]
    u_idx = context["u_index"]
    coord_names = context["coord_names"]
    res_acts = context["residuals"]
    muscles = model.getMuscles()
    m_names = context["muscle_names"]
    forces = context["forces"]
    t = float(context["times"][row])

    qk, u, udot = simbody_kinematics_at_spline_time(
        splines,
        ik.coordinates,
        len(ik.coordinates),
        t,
    )
    state.setTime(t)
    for name, val, spd in zip(ik.coordinates, qk, u, strict=True):
        c = model.getCoordinateSet().get(name)
        c.setValue(state, float(val), False)
        c.setSpeedValue(state, float(spd))
    for m_i, m_name in enumerate(m_names):
        _override(muscles.get(m_name), state, float(forces[row, m_i]))
    for act in res_acts:
        _override(act, state, 0.0)
    model.realizeDynamics(state)

    simbody_udot = osim.Vector(state.getNU(), 0.0)
    for name, acc in zip(ik.coordinates, udot, strict=True):
        simbody_udot.set(u_idx[name], float(acc))

    tau = osim.InverseDynamicsSolver(model).solve(state, simbody_udot)
    for name, act in zip(coord_names, res_acts, strict=True):
        _override(act, state, tau.get(u_idx[name]))

    model.realizeAcceleration(state)
    acc_err = record_max_udot_error(
        context,
        row=row,
        time_s=t,
        coordinates=ik.coordinates,
        u_index=u_idx,
        expected_udot=udot,
        actual_udot=state.getUDot(),
    )
    return udot, acc_err, float(np.linalg.norm(udot))


def collect_frame_reaction_loads(
    context: dict[str, Any],
    state: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Ground-child reactions for L5/S1, knees, patellae, and shoulder norms."""
    f_back_g, _ = _child_reaction(context["joints"]["back"], state)
    R_pelvis = transform_to_matrix(context["pelvis_body"].getTransformInGround(state))[
        :3, :3
    ]
    f_l5s1 = R_pelvis.T @ f_back_g
    R_torso = transform_to_matrix(context["torso_body"].getTransformInGround(state))[
        :3, :3
    ]
    f_l5s1_torso = R_torso.T @ f_back_g

    f_kr, _ = _child_reaction(context["joints"]["knee_r"], state)
    f_kl, _ = _child_reaction(context["joints"]["knee_l"], state)
    f_pr, _ = _child_reaction(context["joints"]["pat_r"], state)
    f_pl, _ = _child_reaction(context["joints"]["pat_l"], state)
    f_sr, _ = _child_reaction(context["joints"]["sh_r"], state)
    f_sl, _ = _child_reaction(context["joints"]["sh_l"], state)

    return (
        f_l5s1,
        f_l5s1_torso,
        f_kr,
        f_kl,
        f_pr,
        f_pl,
        np.array([np.linalg.norm(f_sr), np.linalg.norm(f_sl)]),
    )
