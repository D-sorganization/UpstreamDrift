"""Joint reaction loads for the barbell lifts (OSL-5, UpstreamDrift #13043).

Computes 3D joint reaction forces and moments for the barbell lifts driven by
OSL-4 static-optimization muscle forces and inverse-dynamics reserve torques.
Reuses the OpenSim JointReaction semantics established in the segmental spine
analysis (:mod:`.spine.joint_reactions`).
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
from src.engines.physics_engines.opensim.python.msk_lift_joint_reaction_bound import (
    build_real_lift_bound_record,
    build_real_squat_bound_record,
    mass_above_l5s1,
    muscles_crossing_joint,
    prepare_reaction_model,
    static_hold_analytic_bound,
)
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
from src.shared.python.lifting.bundles import SAME_INPUT_LIFT_BUNDLES
from src.shared.python.lifting.pack_audit.names import LIFTS

logger = logging.getLogger(__name__)

__all__ = [
    "crosscheck_movement_optimizer_l5s1",
    "evaluate_lift_joint_reactions",
    "static_hold_analytic_bound",
]


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

    coord_names = prepare_reaction_model(model)
    return model, coord_names


def _solve_frame_reaction(
    model: Any,
    state: Any,
    context: dict[str, Any],
    row: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
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
        acc_err,
        float(np.linalg.norm(udotk)),
    )


def _compute_reactions_timeseries(
    model: Any,
    coord_names: list[str],
    ik: ikid.IKResult,
    targets: Any,
    muscle_names: list[str],
    forces: np.ndarray,
) -> tuple[JointReactionSeries, np.ndarray, np.ndarray]:
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
        "torso_body": model.getBodySet().get("torso"),
    }

    n = len(targets.time_s)
    c, ap, ml, res = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    c_torso, ap_torso = np.zeros(n), np.zeros(n)
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
    qdd_norms = np.zeros(n)

    for k in range(n):
        (
            f_l5s1,
            f_l5s1_torso,
            f_kr,
            f_kl,
            f_pr,
            f_pl,
            gh_norms,
            err,
            qdd_norm,
        ) = _solve_frame_reaction(model, state, context, k)
        ap[k], c[k], ml[k] = f_l5s1[0], f_l5s1[1], f_l5s1[2]
        c_torso[k], ap_torso[k] = f_l5s1_torso[1], f_l5s1_torso[0]
        res[k] = float(np.linalg.norm(f_l5s1))
        tf_r[k], tf_l[k] = float(np.linalg.norm(f_kr)), float(np.linalg.norm(f_kl))
        pf_r[k], pf_l[k] = float(np.linalg.norm(f_pr)), float(np.linalg.norm(f_pl))
        gh_r[k], gh_l[k] = float(gh_norms[0]), float(gh_norms[1])
        max_err = max(max_err, err)
        qdd_norms[k] = qdd_norm

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
        l5s1_torso_comp_n=c_torso,
        l5s1_torso_ap_shear_n=ap_torso,
        max_acc_error=max_err,
    )
    return series, torso_angles, qdd_norms


OSL4_RECEIPT = (
    Path(__file__).resolve().parents[5]
    / "docs/development/lifting/evidence/opensim_muscles/receipts.json"
)


def _read_osl4_qualification(
    lift: str | Path = "squat",
    path: Path = OSL4_RECEIPT,
) -> dict[str, Any]:
    """Lift qualification copied from the OSL-4 muscle receipt.

    Every value comes from the receipt. A missing receipt or lift record
    yields ``None`` values and a ``reason``, never a default.
    """
    if isinstance(lift, Path):
        path = lift
        lift_name = "squat"
    else:
        lift_name = lift

    keys = (
        "classification",
        "reserve_share",
        "mean_reserve_share",
        "peak_reserve_share",
        "reserve_share_threshold",
    )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **dict.fromkeys(keys),
            "status": "unavailable",
            "reason": f"OSL-4 receipt unreadable: {exc}",
        }
    lift_data = data.get("lifts", {}).get(lift_name)
    if not isinstance(lift_data, dict):
        return {
            **dict.fromkeys(keys),
            "status": "unavailable",
            "reason": f"no {lift_name} record in {path.name}",
        }
    status = lift_data.get("status")
    record = {key: lift_data.get(key) for key in keys}
    record["status"] = status
    if status == "unavailable":
        record["reason"] = lift_data.get(
            "reason", f"{lift_name} is unavailable in OSL-4 receipt"
        )
    if record["reserve_share_threshold"] is None:
        record["reserve_share_threshold"] = data.get("reserve_share_threshold")
    try:
        record["source"] = str(path.relative_to(path.parents[5]))
    except (ValueError, IndexError):
        record["source"] = str(path)
    return record


def _min_acceleration_bound(
    model: Any,
    targets: Any,
    reactions: Any,
    qdd_norms: np.ndarray,
    muscle_names: list[str],
    forces: np.ndarray,
    lift: str = "squat",
) -> tuple[float, dict[str, Any]]:
    """Mass above L5/S1 and the bound record at the least-accelerated frame."""
    min_k = int(np.argmin(qdd_norms))
    m_above = mass_above_l5s1(model)
    w_above_val = m_above * GRAVITY_FLOAT

    crossing = muscles_crossing_joint(model, "back")
    crossing_force = float(
        sum(forces[min_k, muscle_names.index(m)] for m in crossing if m in muscle_names)
    )
    real_lift_info = build_real_lift_bound_record(
        frame_idx=min_k,
        time_s=float(targets.time_s[min_k]),
        qdd_norm=float(qdd_norms[min_k]),
        w_above=w_above_val,
        c_pelvis=float(reactions.l5s1_comp_n[min_k]),
        c_torso=float(reactions.l5s1_torso_comp_n[min_k]),
        crossing_force=crossing_force,
        crossing_count=len(crossing),
        lift=lift,
    )

    return float(m_above), real_lift_info


def evaluate_lift_joint_reactions(
    lift: str,
    *,
    stride: int = 5,
    scratch_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute 3D joint reaction loads for *lift*.

    Generic across all barbell lifts. Lifts lacking an OSL-4 muscle solve
    return an honest ``unavailable`` record explaining why.
    """
    require(lift in LIFTS, f"unknown lift {lift!r}; choose from {list(LIFTS)}")

    osl4_qual = _read_osl4_qualification(lift)
    if osl4_qual.get("status") == "unavailable":
        return unavailable_joint_reaction_record(
            lift,
            osl4_qual.get("reason", f"no OSL-4 muscle solve for {lift}"),
        )
    if osl4_qual.get("mean_reserve_share") is None and "reason" in osl4_qual:
        return unavailable_joint_reaction_record(
            lift,
            osl4_qual["reason"],
        )

    bundle_path = SAME_INPUT_LIFT_BUNDLES.get(lift)
    if bundle_path is None or not bundle_path.is_file():
        return unavailable_joint_reaction_record(
            lift,
            f"no LIFT-5 same-input bundle is registered for {lift!r} in "
            "SAME_INPUT_LIFT_BUNDLES (#11745), so there is no input to fit",
        )

    if not is_engine_available("opensim"):
        return unavailable_joint_reaction_record(
            lift,
            "Native OpenSim runtime is not available in current environment",
        )

    targets, build, ik, inverse = _run_lift_ik_id(lift, stride)
    tmp = scratch_dir or Path(
        tempfile.mkdtemp(prefix=f"opensim_lift_reactions_{lift}_")
    )
    model_path, _, loads_xml = _prepare_scratch_assets(
        lift, build.model, ik, targets, tmp
    )

    _make_muscles_rigid_tendon(build.model)
    basis = MuscleBasis(build.model, ik.coordinates)
    leg_coords = leg_coordinates()
    leg_indices = [ik.coordinates.index(c) for c in leg_coords]
    solve = _solve_lift_frames(basis, ik, inverse, leg_indices, targets.time_s)

    jr_model, coord_names = _setup_joint_reaction_model(model_path, loads_xml)
    reactions, torso_angles, qdd_norms = _compute_reactions_timeseries(
        jr_model, coord_names, ik, targets, basis.muscle_names, solve.forces
    )

    m_above, real_lift_info = _min_acceleration_bound(
        build.model,
        targets,
        reactions,
        qdd_norms,
        basis.muscle_names,
        solve.forces,
        lift=lift,
    )

    bw_n = targets.body_mass_kg * GRAVITY_FLOAT
    phase_segs = detect_motion_phases(lift, targets.time_s, ik.coordinates, ik.q)
    phases = summarize_phase_reactions(phase_segs, targets.time_s, reactions, bw_n)
    cross_check = (
        crosscheck_movement_optimizer_l5s1(
            targets,
            torso_angles,
            (reactions.l5s1_torso_comp_n, reactions.l5s1_torso_ap_shear_n),
            opensim_m_above_kg=float(m_above),
            lift=lift,
        )
        if lift in ("squat", "deadlift")
        else None
    )
    static_bound = static_hold_analytic_bound(lift, build.model, real_lift_info)

    lumbar_idx = (
        ik.coordinates.index("lumbar_extension")
        if "lumbar_extension" in ik.coordinates
        else None
    )
    peak_lumbar = (
        float(np.max(np.abs(inverse.tau[:, lumbar_idx])))
        if lumbar_idx is not None
        else None
    )
    extra_ctx = {
        "bar_mass_kg": targets.bar_mass_kg,
        "peak_lumbar_moment_nm": peak_lumbar,
        "qualification_info": osl4_qual,
    }

    return format_joint_reaction_solution(
        lift,
        targets.time_s,
        bw_n,
        reactions,
        phases,
        cross_check,
        static_bound,
        extra_ctx,
    )
