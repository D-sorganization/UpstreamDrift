"""Muscle forces and activations for the barbell lifts (OSL-4, UpstreamDrift #13042).

Computes muscle forces, activations, and reserve shares for the five barbell
lifts using OpenSim ``StaticOptimization`` and ``MocoInverse`` (where converging),
reusing the existing reserve-share formulation and solvers in
``musculoskeletal_static_opt.py`` and ``musculoskeletal_solvers.py``.

For the back squat (the only lift with a credible same-input bundle and passing
Hicks/marker residual gates today), the pipeline solves per-frame muscle
redundancy across the 80 lower-limb muscles, verifies joint-moment reconstruction
against inverse dynamics, and evaluates reserve share.  The other four lifts
(bench press, deadlift, snatch, clean and jerk) are honestly recorded as
``unavailable`` with their blocking reasons, never as zero.

Major functional muscle groups reported:
- quadriceps (8 muscles)
- gluteals (18 muscles)
- hamstrings (8 muscles)
- erector spinae (0 muscles in base Rajagopal lower-limb model)
- deltoids (0 muscles in base Rajagopal lower-limb model)
- trapezius (0 muscles in base Rajagopal lower-limb model)
- triceps (0 muscles in base Rajagopal lower-limb model)

Model muscle list citation:
    Rajagopal A, Dembia CL, DeMers MS, Delp DD, Hicks JL, Delp SL.
    Full-Body Musculoskeletal Model for Muscle-Driven Simulation of Human Gait.
    IEEE Trans Biomed Eng. 2016;63(10):2068-2079. doi:10.1109/TBME.2016.2586862
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
from src.engines.physics_engines.opensim.python.msk_lift_arm_reach import (
    reaching_lift_build,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_graft import SIDES
from src.engines.physics_engines.opensim.python.musculoskeletal_solvers import (
    SolveWindow,
    solve_moco_pilot,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_static_opt import (
    LEG_JOINTS,
    MuscleBasis,
    leg_coordinates,
    solve_frame,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
    RESERVE_OPTIMAL_FORCE,
    UNLOCKED_COORDINATES,
    _replace_actuators,
    resolve_base_model,
    write_sto,
)
from src.shared.python.contracts import require
from src.shared.python.engine_core.engine_availability import is_engine_available
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.lifting.bundle_targets import LiftTargets, lift_targets
from src.shared.python.lifting.bundles import SAME_INPUT_LIFT_BUNDLES
from src.shared.python.lifting.lift_spec import lift_phases
from src.shared.python.lifting.pack_audit.names import LIFTS

logger = logging.getLogger(__name__)

__all__ = [
    "GROUP_TO_RAJAGOPAL_MUSCLES",
    "MAJOR_MUSCLE_GROUPS",
    "RAJAGOPAL_MODEL_CITATION",
    "RAJAGOPAL_MUSCLE_TO_GROUP",
    "RESERVE_SHARE_THRESHOLD",
    "classify_lift_reserves",
    "evaluate_lift_muscles",
    "reserve_share",
]

#: Model citation for the Rajagopal 2016 / 2023 lower-limb muscle list.
RAJAGOPAL_MODEL_CITATION: str = (
    "Rajagopal A, Dembia CL, DeMers MS, Delp DD, Hicks JL, Delp SL. "
    "Full-Body Musculoskeletal Model for Muscle-Driven Simulation of Human Gait. "
    "IEEE Trans Biomed Eng. 2016;63(10):2068-2079. doi:10.1109/TBME.2016.2586862"
)

#: Threshold for reserve actuator moment share (5%). Below this is qualified.
RESERVE_SHARE_THRESHOLD: float = 0.05

#: The 7 major muscle groups required by OSL-4.
MAJOR_MUSCLE_GROUPS: tuple[str, ...] = (
    "quadriceps",
    "gluteals",
    "hamstrings",
    "erector spinae",
    "deltoids",
    "trapezius",
    "triceps",
)

#: Mapping from major functional group to constituent Rajagopal muscle names.
#: Note: Rajagopal et al. (2016) is an 80-muscle lower-extremity model.
#: Erector spinae, deltoids, trapezius, and triceps have no muscles in this
#: base model; their joint moments are carried by upper-body coordinate actuators.
GROUP_TO_RAJAGOPAL_MUSCLES: dict[str, tuple[str, ...]] = {
    "quadriceps": (
        "recfem_r",
        "recfem_l",
        "vasint_r",
        "vasint_l",
        "vaslat_r",
        "vaslat_l",
        "vasmed_r",
        "vasmed_l",
    ),
    "gluteals": (
        "glmax1_r",
        "glmax1_l",
        "glmax2_r",
        "glmax2_l",
        "glmax3_r",
        "glmax3_l",
        "glmed1_r",
        "glmed1_l",
        "glmed2_r",
        "glmed2_l",
        "glmed3_r",
        "glmed3_l",
        "glmin1_r",
        "glmin1_l",
        "glmin2_r",
        "glmin2_l",
        "glmin3_r",
        "glmin3_l",
    ),
    "hamstrings": (
        "bflh_r",
        "bflh_l",
        "bfsh_r",
        "bfsh_l",
        "semimem_r",
        "semimem_l",
        "semiten_r",
        "semiten_l",
    ),
    "erector spinae": (),
    "deltoids": (),
    "trapezius": (),
    "triceps": (),
}

#: Reverse mapping: Rajagopal muscle name -> major functional group.
RAJAGOPAL_MUSCLE_TO_GROUP: dict[str, str] = {
    muscle: group
    for group, muscles in GROUP_TO_RAJAGOPAL_MUSCLES.items()
    for muscle in muscles
}


def reserve_share(tau: np.ndarray, reserves: np.ndarray) -> float:
    """Calculate reserve actuator share: ``sum |r| / sum |tau|``.

    Returns 0.0 when reserves are zero; returns infinity when reserves are
    non-zero on zero joint moment demand.
    """
    total = float(np.sum(np.abs(tau)))
    used = float(np.sum(np.abs(reserves)))
    if used == 0.0:
        return 0.0
    return used / total if total > 1e-12 else math.inf


def classify_lift_reserves(
    share: float, threshold: float = RESERVE_SHARE_THRESHOLD
) -> str:
    """Classify a lift as ``qualified`` or ``unqualified``.

    A lift is qualified only when reserve share is strictly below threshold.
    """
    require(threshold > 0.0, "threshold must be positive")
    return "qualified" if share < threshold else "unqualified"


def _make_muscles_rigid_tendon(model: Any) -> None:
    """Set ignore_tendon_compliance to True on all muscles for static optimization."""
    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        muscles.get(i).set_ignore_tendon_compliance(True)


def _prepare_actuated_model(build_model: Any) -> Any:
    """Clone build model, unlock coordinates, and add reserve/upper actuators."""
    osim = import_opensim()
    model = osim.Model(build_model)
    _make_muscles_rigid_tendon(model)
    for name in UNLOCKED_COORDINATES:
        coord = model.updCoordinateSet().get(name)
        coord.set_locked(False)
        coord.setDefaultLocked(False)
    for i in range(model.getCoordinateSet().getSize()):
        coord = model.updCoordinateSet().get(i)
        if coord.getName() not in ("mtp_angle_r", "mtp_angle_l"):
            coord.set_locked(False)
            coord.setDefaultLocked(False)
    ikid._massless_bar(model)
    _replace_actuators(model, RESERVE_OPTIMAL_FORCE)
    return model


def _build_external_loads(
    targets: LiftTargets, model: Any, scratch_dir: Path
) -> tuple[Path, Path]:
    """Write external loads .sto and .xml for ground reaction and barbell forces."""
    osim = import_opensim()
    times = targets.time_s
    n_frames = len(times)

    loads_data: dict[str, np.ndarray] = {}
    ext_loads = osim.ExternalLoads()

    # Feet ground reaction forces
    for side, calcn in (("left", "calcn_l"), ("right", "calcn_r")):
        for ax, comp in zip("xyz", range(3), strict=False):
            loads_data[f"{calcn}_force_v{ax}"] = targets.grf_force_n[side][:, comp]
            loads_data[f"{calcn}_torque_{ax}"] = targets.grf_moment_nm[side][:, comp]
            loads_data[f"{calcn}_point_p{ax}"] = np.zeros(n_frames)
        ef = osim.ExternalForce()
        ef.setName(f"{calcn}_grf")
        ef.set_applied_to_body(calcn)
        ef.set_force_expressed_in_body("ground")
        ef.set_point_expressed_in_body(calcn)
        ef.set_force_identifier(f"{calcn}_force_v")
        ef.set_point_identifier(f"{calcn}_point_p")
        ef.set_torque_identifier(f"{calcn}_torque_")
        ext_loads.cloneAndAppend(ef)

    # Hand grip loads
    state = model.initSystem()
    for side_code, hand_body in (("L", "hand_l"), ("R", "hand_r")):
        body = model.getBodySet().get(hand_body)
        grip_pt = [float(body.getMassCenter().get(i)) for i in range(3)]
        for ax, comp in zip("xyz", range(3), strict=False):
            loads_data[f"{hand_body}_force_v{ax}"] = targets.hand_force_n[side_code][
                :, comp
            ]
            loads_data[f"{hand_body}_torque_{ax}"] = targets.hand_torque_nm[side_code][
                :, comp
            ]
            loads_data[f"{hand_body}_point_p{ax}"] = np.full(n_frames, grip_pt[comp])
        ef_hand = osim.ExternalForce()
        ef_hand.setName(f"{hand_body}_bar")
        ef_hand.set_applied_to_body(hand_body)
        ef_hand.set_force_expressed_in_body("ground")
        ef_hand.set_point_expressed_in_body(hand_body)
        ef_hand.set_force_identifier(f"{hand_body}_force_v")
        ef_hand.set_point_identifier(f"{hand_body}_point_p")
        ef_hand.set_torque_identifier(f"{hand_body}_torque_")
        ext_loads.cloneAndAppend(ef_hand)

    sto_path = scratch_dir / "external_loads.sto"
    write_sto(sto_path, times, loads_data)
    ext_loads.setDataFileName(str(sto_path.resolve()))
    xml_path = scratch_dir / "external_loads.xml"
    ext_loads.printToXML(str(xml_path))
    return sto_path, xml_path


def _summarize_groups(
    activations: np.ndarray, forces: np.ndarray, muscle_names: list[str]
) -> dict[str, dict[str, Any]]:
    """Summarize peak and mean activation and force per major group."""
    name_to_idx = {name: i for i, name in enumerate(muscle_names)}
    summary: dict[str, dict[str, Any]] = {}
    for group_name in MAJOR_MUSCLE_GROUPS:
        constituent = GROUP_TO_RAJAGOPAL_MUSCLES[group_name]
        indices = [name_to_idx[m] for m in constituent if m in name_to_idx]
        if not indices:
            summary[group_name] = {
                "n_muscles": 0,
                "peak_activation": None,
                "mean_activation": None,
                "peak_force_n": None,
                "mean_force_n": None,
            }
            continue
        g_act = activations[:, indices]
        g_frc = forces[:, indices]
        summary[group_name] = {
            "n_muscles": len(indices),
            "peak_activation": float(np.max(g_act)),
            "mean_activation": float(np.mean(g_act)),
            "peak_force_n": float(np.max(g_frc)),
            "mean_force_n": float(np.mean(g_frc)),
        }
    return summary


def _summarize_phases(
    times: np.ndarray,
    activations: np.ndarray,
    forces: np.ndarray,
    muscle_names: list[str],
    lift: str,
) -> dict[str, dict[str, Any]]:
    """Summarize muscle group metrics per lift phase."""
    phases = lift_phases(lift)
    t_start, t_end = float(times[0]), float(times[-1])
    duration = t_end - t_start
    phase_summary: dict[str, dict[str, Any]] = {}

    for i, phase in enumerate(phases):
        t_phase_start = t_start + phase.t_frac * duration
        if i + 1 < len(phases):
            t_phase_end = t_start + phases[i + 1].t_frac * duration
            mask = (times >= t_phase_start) & (times < t_phase_end)
        else:
            mask = times >= t_phase_start
        if not np.any(mask):
            continue
        phase_act = activations[mask]
        phase_frc = forces[mask]
        phase_summary[phase.name] = _summarize_groups(
            phase_act, phase_frc, muscle_names
        )

    return phase_summary


def _run_moco_eval(
    model_path: Path, coords_path: Path, loads_xml: Path, times: np.ndarray
) -> dict[str, Any]:
    """Run honest MocoInverse pilot assessment."""
    try:
        window = SolveWindow(float(times[0]), float(times[-1]))
        return solve_moco_pilot(
            model_path,
            coords_path,
            loads_xml,
            window,
            mesh_interval=0.02,
            max_iterations=20,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Moco pilot failed: %s", exc)
        return {
            "solver": "opensim.MocoInverse",
            "converged": False,
            "status": f"Failed: {exc}",
            "error": str(exc),
        }


def evaluate_lift_muscles(
    lift: str,
    *,
    stride: int = 5,
    scratch_dir: Path | None = None,
    run_moco: bool = True,
) -> dict[str, Any]:
    """Compute muscle activations, forces, and reserve shares for *lift*.

    Generic across all barbell lifts.  For lifts lacking an IK/ID bundle,
    returns an honest ``unavailable`` record.
    """
    require(lift in LIFTS, f"unknown lift {lift!r}; choose from {list(LIFTS)}")

    if lift != "squat":
        return {
            "lift": lift,
            "status": "unavailable",
            "reason": (
                f"no LIFT-5 same-input bundle is registered for {lift!r} in "
                "SAME_INPUT_LIFT_BUNDLES (#11745), so there is no input to fit"
            ),
            "classification": "unqualified",
            "reserve_share": None,
            "mean_reserve_share": None,
            "peak_reserve_share": None,
            "forces": None,
            "activations": None,
            "major_groups": None,
            "phases": None,
            "moco": None,
        }

    if not is_engine_available("opensim"):
        return {
            "lift": lift,
            "status": "unavailable",
            "reason": "Native OpenSim runtime is not available in current environment",
            "classification": "unqualified",
            "reserve_share": None,
            "mean_reserve_share": None,
            "peak_reserve_share": None,
            "forces": None,
            "activations": None,
            "major_groups": None,
            "phases": None,
            "moco": None,
        }

    path = SAME_INPUT_LIFT_BUNDLES[lift]
    assert path is not None

    targets = lift_targets(lift, path, stride=stride)
    base_model = resolve_base_model()
    scales = ikid.bundle_segment_scales(targets, base_model)
    build, reach = reaching_lift_build(lift, targets, base_model, scales)
    pelvis = ikid.pelvis_landmark_targets(build, targets, base_model)
    ik = ikid.run_inverse_kinematics(build, targets, pelvis)
    inverse = ikid.run_inverse_dynamics(build, targets, ik)

    # Set up scratch directory for serialized assets
    work_dir = scratch_dir or Path("/tmp/opensim_lift_muscles_squat")
    work_dir.mkdir(parents=True, exist_ok=True)

    actuated_model = _prepare_actuated_model(build.model)
    model_path = work_dir / "squat_msk.osim"
    actuated_model.printToXML(str(model_path))

    coords_dict = {name: ik.q[:, j] for j, name in enumerate(ik.coordinates)}
    coords_path = work_dir / "coordinates.sto"
    write_sto(coords_path, targets.time_s, coords_dict)

    _, loads_xml = _build_external_loads(targets, actuated_model, work_dir)

    # Setup MuscleBasis with rigid-tendon Millard muscles
    _make_muscles_rigid_tendon(build.model)
    basis = MuscleBasis(build.model, ik.coordinates)
    leg_coords = leg_coordinates()
    leg_indices = [ik.coordinates.index(c) for c in leg_coords]

    n_frames = len(targets.time_s)
    n_muscles = len(basis.muscles)
    n_leg_coords = len(leg_coords)

    activations = np.zeros((n_frames, n_muscles))
    forces = np.zeros((n_frames, n_muscles))
    reserves = np.zeros((n_frames, n_leg_coords))
    tau_muscles = np.zeros((n_frames, n_leg_coords))
    tau_targets = np.zeros((n_frames, n_leg_coords))
    moment_balance_errors = np.zeros(n_frames)

    v_coords = np.gradient(ik.q, targets.time_s, axis=0)

    for i in range(n_frames):
        q_frame = ik.q[i]
        v_frame = v_coords[i]
        tau_leg = inverse.tau[i, leg_indices]
        tau_targets[i] = tau_leg

        active, passive, moment = basis.evaluate(q_frame, v_frame)
        sol = solve_frame(active, passive, moment, tau_leg)

        activations[i] = sol.activation
        forces[i] = passive + sol.activation * active
        reserves[i] = sol.reserve
        tau_m = moment.T @ forces[i]
        tau_muscles[i] = tau_m

        # Moment balance residual: tau_m + r == tau
        residual = tau_m + sol.reserve - tau_leg
        moment_balance_errors[i] = float(np.max(np.abs(residual)))

    frame_reserve_shares = [
        reserve_share(tau_targets[i], reserves[i]) for i in range(n_frames)
    ]
    mean_share = float(np.mean(frame_reserve_shares))
    peak_share = float(np.max(frame_reserve_shares))
    total_share = float(np.sum(np.abs(reserves)) / np.sum(np.abs(tau_targets)))
    classification = classify_lift_reserves(total_share)

    major_groups = _summarize_groups(activations, forces, basis.muscle_names)
    phases = _summarize_phases(
        targets.time_s, activations, forces, basis.muscle_names, lift
    )

    moco_record = (
        _run_moco_eval(model_path, coords_path, loads_xml, targets.time_s)
        if run_moco
        else {
            "solver": "opensim.MocoInverse",
            "converged": False,
            "status": "skipped",
        }
    )

    return {
        "lift": lift,
        "status": "complete",
        "reason": "",
        "classification": classification,
        "reserve_share": total_share,
        "mean_reserve_share": mean_share,
        "peak_reserve_share": peak_share,
        "moment_balance_max_error_nm": float(np.max(moment_balance_errors)),
        "times": targets.time_s,
        "muscle_names": basis.muscle_names,
        "leg_coordinates": leg_coords,
        "activations": activations,
        "forces": forces,
        "reserves": reserves,
        "tau_targets": tau_targets,
        "tau_muscles": tau_muscles,
        "major_groups": major_groups,
        "phases": phases,
        "moco": moco_record,
        "citation": RAJAGOPAL_MODEL_CITATION,
    }
