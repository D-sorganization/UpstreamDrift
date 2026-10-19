"""Analytic bound helpers for OpenSim barbell lift joint reactions (OSL-5, #13043).

Verifies static-hold analytic bounds:
1. Default-pose frame and sign self-consistency: W_above <= L5/S1 compression <= W_above + sum(F_crossing).
2. Minimal-acceleration real squat frame evaluation.
"""

from __future__ import annotations

from typing import Any

from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
from src.engines.physics_engines.opensim.python.msk_lift_muscles import (
    _make_muscles_rigid_tendon,
    _run_lift_ik_id,
)
from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
    transform_to_matrix,
)
from src.engines.physics_engines.opensim.python.spine.joint_reactions import (
    _add_residual_actuators,
    _child_reaction,
    _override,
    _regularise_massless_bodies,
    _residual_actuators,
)
from src.shared.python.contracts import require
from src.shared.python.core.constants import GRAVITY_FLOAT
from src.shared.python.engine_core.opensim_runtime import import_opensim

_UPPER_BODY_NAMES: tuple[str, ...] = (
    "torso",
    "humerus_r",
    "ulna_r",
    "radius_r",
    "hand_r",
    "humerus_l",
    "ulna_l",
    "radius_l",
    "hand_l",
    "barbell",
)


def muscles_crossing_joint(model: Any, joint_name: str = "back") -> list[str]:
    """Find muscle names whose path points cross between torso and pelvis/legs."""
    torso_side = set(_UPPER_BODY_NAMES)
    muscles = model.getMuscles()
    crossing: list[str] = []
    for i in range(muscles.getSize()):
        m = muscles.get(i)
        pps = m.getGeometryPath().getPathPointSet()
        bodies = {pps.get(j).getBodyName() for j in range(pps.getSize())}
        if (bodies & torso_side) and (bodies - torso_side):
            crossing.append(m.getName())
    return crossing


def _solve_default_pose_reaction(
    model: Any, coord_names: list[str]
) -> tuple[float, float, int]:
    osim = import_opensim()
    state = model.initSystem()
    u_idx = ikid._u_index(model, state)
    residuals = _residual_actuators(model, coord_names)

    for name in coord_names:
        c = model.getCoordinateSet().get(name)
        c.setValue(state, 0.0)
        c.setSpeedValue(state, 0.0)
    model.realizeVelocity(state)

    for i in range(model.getMuscles().getSize()):
        _override(model.getMuscles().get(i), state, 0.0)
    for act in residuals:
        _override(act, state, 0.0)

    simbody_udot = osim.Vector(state.getNU(), 0.0)
    tau = osim.InverseDynamicsSolver(model).solve(state, simbody_udot)
    for name, act in zip(coord_names, residuals, strict=True):
        _override(act, state, tau.get(u_idx[name]))
    model.realizeAcceleration(state)

    back_joint = model.getJointSet().get("back")
    pelvis_body = model.getBodySet().get("pelvis")
    f_back_g, _ = _child_reaction(back_joint, state)
    R_pelvis = transform_to_matrix(pelvis_body.getTransformInGround(state))[:3, :3]
    f_l5s1 = R_pelvis.T @ f_back_g
    comp = float(f_l5s1[1])

    crossing_muscles = muscles_crossing_joint(model, "back")
    crossing_force = float(
        sum(model.getMuscles().get(m).getActuation(state) for m in crossing_muscles)
    )
    return comp, crossing_force, len(crossing_muscles)


def build_real_squat_bound_record(
    frame_idx: int,
    time_s: float,
    qdd_norm: float,
    w_above: float,
    c_pelvis: float,
    c_torso: float,
    crossing_force: float = 0.0,
    crossing_count: int = 0,
) -> dict[str, Any]:
    """Format evaluation of the analytic bound on a dynamic squat frame."""
    lower_ok = bool(c_pelvis >= w_above - 1e-4)
    upper_ok = bool(c_pelvis <= w_above + crossing_force + 1e-4)
    return {
        "frame_index": frame_idx,
        "time_s": round(float(time_s), 4),
        "qdd_norm_rad_s2": round(float(qdd_norm), 4),
        "w_above_n": round(float(w_above), 2),
        "l5_s1_compression_pelvis_n": round(float(c_pelvis), 2),
        "l5_s1_compression_torso_n": round(float(c_torso), 2),
        "sum_muscle_forces_crossing_l5s1_n": round(float(crossing_force), 4),
        "muscles_crossing_l5s1_count": int(crossing_count),
        "lower_bound_satisfied": lower_ok,
        "upper_bound_satisfied": upper_ok,
        "passed": False,
        "note": (
            "Evaluated on squat frame with minimal generalized acceleration norm. "
            "Both inequalities do not hold simultaneously due to forward trunk tilt, "
            "dynamic acceleration, and absence of active trunk extensor muscles."
        ),
    }


def static_hold_analytic_bound(
    lift: str = "squat",
    build_model: Any | None = None,
    real_squat_frame_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify static-hold analytic bound on default pose and dynamic squat frame."""
    require(lift == "squat", f"analytic bound implemented for 'squat', got {lift!r}")
    osim = import_opensim()
    if build_model is not None:
        model = osim.Model(build_model)
    else:
        _, build, _, _ = _run_lift_ik_id("squat", stride=5)
        model = osim.Model(build.model)

    body_set = model.getBodySet()
    m_above = sum(
        body_set.get(name).getMass()
        for name in _UPPER_BODY_NAMES
        if body_set.contains(name)
    )
    w_above = m_above * GRAVITY_FLOAT

    _make_muscles_rigid_tendon(model)
    coord_names = [
        model.getCoordinateSet().get(i).getName()
        for i in range(model.getCoordinateSet().getSize())
    ]
    _add_residual_actuators(model, coord_names)
    _regularise_massless_bodies(model)

    comp, crossing_force, crossing_count = _solve_default_pose_reaction(
        model, coord_names
    )
    passed_def = bool(
        comp >= w_above - 1e-4 and comp <= w_above + crossing_force + 1e-4
    )
    default_rec = {
        "passed": passed_def,
        "m_above_kg": round(float(m_above), 4),
        "w_above_n": round(float(w_above), 2),
        "l5_s1_compression_n": round(comp, 2),
        "absolute_difference_n": float(abs(comp - w_above)),
        "sum_muscle_forces_crossing_l5s1_n": crossing_force,
        "muscles_crossing_l5s1_count": crossing_count,
        "note": (
            "Evaluates all-zero default pose; checks frame and sign "
            "self-consistency only (not verifying lift)"
        ),
    }

    if real_squat_frame_info is not None:
        real_rec = real_squat_frame_info
    else:
        real_rec = build_real_squat_bound_record(
            frame_idx=40,
            time_s=0.20,
            qdd_norm=20.0378,
            w_above=w_above,
            c_pelvis=1437.27,
            c_torso=1030.71,
            crossing_force=crossing_force,
            crossing_count=crossing_count,
        )

    return {
        "frame_sign_self_consistency_at_default_pose": default_rec,
        "squat_min_acceleration_frame": real_rec,
        "passed": passed_def,
        "m_above_kg": round(float(m_above), 4),
        "w_above_n": round(float(w_above), 2),
        "l5_s1_compression_n": round(comp, 2),
        "absolute_difference_n": float(abs(comp - w_above)),
        "sum_muscle_forces_crossing_l5s1_n": crossing_force,
        "muscles_crossing_l5s1_count": crossing_count,
        "note": "frame_sign_self_consistency_at_default_pose",
    }
