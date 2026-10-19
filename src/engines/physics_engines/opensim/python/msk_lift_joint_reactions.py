"""Joint reaction loads for the barbell lifts (OSL-5, UpstreamDrift #13043).

Computes 3D joint reaction forces and moments for the barbell lifts driven by
OSL-4 static-optimization muscle forces and inverse-dynamics reserve torques.
Reuses the OpenSim JointReaction semantics established in the segmental spine
analysis (:mod:`.spine.joint_reactions`).

Lifts evaluated:
- Squat: full joint reaction analysis across L5/S1 (lumbar 'back' joint, pelvis ISB
  frame), knee (tibiofemoral resultant and patellofemoral load), and shoulder
  (glenohumeral 'acromial' joint, articulated in the model).
- Other four lifts (bench press, deadlift, snatch, clean and jerk): honestly recorded
  as ``unavailable`` with blocking reasons ("no OSL-4 muscle solve (needs OSL-3 #13041 bundles)").
"""

from __future__ import annotations

import logging
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
from src.engines.physics_engines.opensim.python.msk_lift_joint_reaction_report import (
    JointReactionSeries,
    crosscheck_movement_optimizer_l5s1,
    format_joint_reaction_solution,
    summarize_phase_reactions,
    unavailable_joint_reaction_record,
)
from src.engines.physics_engines.opensim.python.msk_lift_muscles import (
    MuscleBasis,
    _make_muscles_rigid_tendon,
    _prepare_scratch_assets,
    _run_lift_ik_id,
    _solve_lift_frames,
    detect_motion_phases,
    leg_coordinates,
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
from src.shared.python.engine_core.engine_availability import is_engine_available
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.lifting.pack_audit.names import LIFTS

logger = logging.getLogger(__name__)

__all__ = [
    "crosscheck_movement_optimizer_l5s1",
    "evaluate_lift_joint_reactions",
    "static_hold_analytic_bound",
]

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


def _setup_joint_reaction_model(
    model_path: Path, loads_xml: Path
) -> tuple[Any, list[str]]:
    osim = import_opensim()
    processor = osim.ModelProcessor(str(model_path))
    processor.append(osim.ModOpAddExternalLoads(str(loads_xml)))
    model = processor.process()
    model.finalizeFromProperties()

    for i in range(model.getMuscles().getSize()):
        model.getMuscles().get(i).set_ignore_tendon_compliance(True)

    coord_names = [
        model.getCoordinateSet().get(i).getName()
        for i in range(model.getCoordinateSet().getSize())
    ]
    _add_residual_actuators(model, coord_names)
    _regularise_massless_bodies(model)
    return model, coord_names


def _solve_frame_reaction(
    model: Any,
    state: Any,
    context: dict[str, Any],
    row: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
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

    qk, uk, udotk = ikid._derivatives(splines, len(ik.coordinates), t)
    for name, val, spd in zip(ik.coordinates, qk, uk, strict=True):
        c = model.getCoordinateSet().get(name)
        c.setValue(state, float(val))
        c.setSpeedValue(state, float(spd))
    model.realizeVelocity(state)

    for m_i, m_name in enumerate(m_names):
        _override(muscles.get(m_name), state, float(forces[row, m_i]))
    for act in res_acts:
        _override(act, state, 0.0)

    simbody_udot = osim.Vector(state.getNU(), 0.0)
    for name, acc in zip(ik.coordinates, udotk, strict=True):
        simbody_udot.set(u_idx[name], float(acc))

    tau = osim.InverseDynamicsSolver(model).solve(state, simbody_udot)
    for name, act in zip(coord_names, res_acts, strict=True):
        _override(act, state, tau.get(u_idx[name]))

    model.realizeAcceleration(state)
    actual_udot = state.getUDot()
    acc_err = max(
        abs(actual_udot.get(u_idx[name]) - acc)
        for name, acc in zip(ik.coordinates, udotk, strict=True)
    )

    f_back_g, _ = _child_reaction(context["joints"]["back"], state)
    R_pelvis = transform_to_matrix(context["pelvis_body"].getTransformInGround(state))[
        :3, :3
    ]
    f_l5s1 = R_pelvis.T @ f_back_g

    f_kr, _ = _child_reaction(context["joints"]["knee_r"], state)
    f_kl, _ = _child_reaction(context["joints"]["knee_l"], state)
    f_pr, _ = _child_reaction(context["joints"]["pat_r"], state)
    f_pl, _ = _child_reaction(context["joints"]["pat_l"], state)
    f_sr, _ = _child_reaction(context["joints"]["sh_r"], state)
    f_sl, _ = _child_reaction(context["joints"]["sh_l"], state)

    return (
        f_l5s1,
        f_kr,
        f_kl,
        f_pr,
        f_pl,
        np.array([np.linalg.norm(f_sr), np.linalg.norm(f_sl)]),
        acc_err,
    )


def _compute_reactions_timeseries(
    model: Any,
    coord_names: list[str],
    ik: ikid.IKResult,
    targets: Any,
    muscle_names: list[str],
    forces: np.ndarray,
) -> tuple[JointReactionSeries, np.ndarray]:
    state = model.initSystem()
    u_idx = ikid._u_index(model, state)
    res_acts = _residual_actuators(model, coord_names)
    splines = ikid._splines(ik, targets.time_s)

    joints = {
        "back": model.getJointSet().get("back"),
        "knee_r": model.getJointSet().get("walker_knee_r"),
        "knee_l": model.getJointSet().get("walker_knee_l"),
        "pat_r": model.getJointSet().get("patellofemoral_r"),
        "pat_l": model.getJointSet().get("patellofemoral_l"),
        "sh_r": model.getJointSet().get("acromial_r"),
        "sh_l": model.getJointSet().get("acromial_l"),
    }
    context = {
        "splines": splines,
        "ik": ik,
        "u_index": u_idx,
        "coord_names": coord_names,
        "residuals": res_acts,
        "muscle_names": muscle_names,
        "forces": forces,
        "times": targets.time_s,
        "joints": joints,
        "pelvis_body": model.getBodySet().get("pelvis"),
    }

    n = len(targets.time_s)
    c, ap, ml, res = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    torso_angles = np.zeros(n)
    torso_body = model.getBodySet().get("torso")
    tf_r, tf_l, pf_r, pf_l, gh_r, gh_l = (
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
    )
    max_err = 0.0

    for k in range(n):
        f_l5s1, f_kr, f_kl, f_pr, f_pl, gh_norms, err = _solve_frame_reaction(
            model, state, context, k
        )
        ap[k], c[k], ml[k] = f_l5s1[0], f_l5s1[1], f_l5s1[2]
        res[k] = float(np.linalg.norm(f_l5s1))
        tf_r[k], tf_l[k] = float(np.linalg.norm(f_kr)), float(np.linalg.norm(f_kl))
        pf_r[k], pf_l[k] = float(np.linalg.norm(f_pr)), float(np.linalg.norm(f_pl))
        gh_r[k], gh_l[k] = float(gh_norms[0]), float(gh_norms[1])
        max_err = max(max_err, err)

        R_t = transform_to_matrix(torso_body.getTransformInGround(state))[:3, :3]
        torso_angles[k] = math.atan2(R_t[0, 1], R_t[1, 1])

    series = JointReactionSeries(
        time_s=targets.time_s,
        l5s1_comp_n=c,
        l5s1_ap_shear_n=ap,
        l5s1_ml_shear_n=ml,
        l5s1_resultant_n=res,
        knee_tf_r_n=tf_r,
        knee_tf_l_n=tf_l,
        knee_pf_r_n=pf_r,
        knee_pf_l_n=pf_l,
        shoulder_gh_r_n=gh_r,
        shoulder_gh_l_n=gh_l,
        max_acc_error=max_err,
    )
    return series, torso_angles


def static_hold_analytic_bound(
    lift: str = "squat", build_model: Any | None = None
) -> dict[str, Any]:
    """Verify static-hold analytic bound: W_above <= L5/S1 compression <= W_above + sum(F_muscles)."""
    require(
        lift == "squat",
        f"analytic bound currently implemented for 'squat', got {lift!r}",
    )
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

    # Crossing muscles in Rajagopal 80-muscle model is 0
    crossing_muscle_force = 0.0
    passed = bool(
        comp >= w_above - 1e-4 and comp <= w_above + crossing_muscle_force + 1e-4
    )
    return {
        "passed": passed,
        "m_above_kg": round(float(m_above), 4),
        "w_above_n": round(float(w_above), 2),
        "l5_s1_compression_n": round(comp, 2),
        "absolute_difference_n": float(abs(comp - w_above)),
        "sum_muscle_forces_crossing_l5s1_n": crossing_muscle_force,
        "muscles_crossing_l5s1_count": 0,
        "note": "Rajagopal 80-muscle model has 0 torso muscles; bound holds with equality in upright static hold",
    }


def evaluate_lift_joint_reactions(
    lift: str,
    *,
    stride: int = 5,
    scratch_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute 3D joint reaction loads for *lift*.

    Generic across all barbell lifts. Non-squat lifts return an honest
    ``unavailable`` record explaining missing OSL-4 muscle solves.
    """
    require(lift in LIFTS, f"unknown lift {lift!r}; choose from {list(LIFTS)}")

    if lift != "squat":
        return unavailable_joint_reaction_record(
            lift,
            "no OSL-4 muscle solve (needs OSL-3 #13041 bundles)",
        )

    if not is_engine_available("opensim"):
        return unavailable_joint_reaction_record(
            lift,
            "Native OpenSim runtime is not available in current environment",
        )

    targets, build, ik, inverse = _run_lift_ik_id(lift, stride)
    tmp = scratch_dir or Path(tempfile.mkdtemp(prefix="opensim_lift_reactions_squat_"))
    model_path, _, loads_xml = _prepare_scratch_assets(build.model, ik, targets, tmp)

    _make_muscles_rigid_tendon(build.model)
    basis = MuscleBasis(build.model, ik.coordinates)
    leg_coords = leg_coordinates()
    leg_indices = [ik.coordinates.index(c) for c in leg_coords]
    solve = _solve_lift_frames(basis, ik, inverse, leg_indices, targets.time_s)

    jr_model, coord_names = _setup_joint_reaction_model(model_path, loads_xml)
    reactions, torso_angles = _compute_reactions_timeseries(
        jr_model, coord_names, ik, targets, basis.muscle_names, solve.forces
    )

    bw_n = targets.body_mass_kg * GRAVITY_FLOAT
    phase_segs = detect_motion_phases(lift, targets.time_s, ik.coordinates, ik.q)
    phases = summarize_phase_reactions(phase_segs, targets.time_s, reactions, bw_n)
    cross_check = crosscheck_movement_optimizer_l5s1(targets, torso_angles)
    static_bound = static_hold_analytic_bound("squat", build.model)

    return format_joint_reaction_solution(
        lift,
        targets.time_s,
        bw_n,
        reactions,
        phases,
        cross_check,
        static_bound,
    )
