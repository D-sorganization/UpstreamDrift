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
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
from src.engines.physics_engines.opensim.python.msk_lift_arm_reach import (
    reaching_lift_build,
)
from src.engines.physics_engines.opensim.python.msk_lift_muscle_report import (
    LiftSolveResults,
    compute_squat_diagnostics,
    format_squat_solution,
    load_ik_id_residual_status,
    unavailable_lift_record,
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
    "detect_motion_phases",
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
    share: float,
    threshold: float = RESERVE_SHARE_THRESHOLD,
    *,
    residual_gate_passed: bool = True,
) -> str:
    """Classify a lift as ``qualified`` or ``unqualified``.

    A lift is qualified only when reserve share is strictly below threshold
    AND the IK/ID residual gate passed.
    """
    require(threshold > 0.0, "threshold must be positive")
    if not residual_gate_passed:
        return "unqualified"
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


def detect_motion_phases(
    lift: str,
    times: np.ndarray,
    coordinate_names: tuple[str, ...] | list[str],
    q: np.ndarray,
) -> list[tuple[str, float, float]]:
    """Derive lift motion segments from kinematics rather than arbitrary window fractions.

    For a squat, standing requires knee flexion < 25 degrees. If the kinematics
    stay in deep flexion (> 45 degrees, e.g. the 0.2 s engine window around 110 degrees),
    this function reports a single honest segment: ``bottom (engine window X.X-Y.Y s)``
    and never falsely labels nonexistent standing, descent, or lockout phases.
    """
    require(len(times) == len(q), "times and q rows must match")
    require(len(times) > 0, "times must not be empty")
    t_start, t_end = float(times[0]), float(times[-1])

    if lift == "squat":
        coords = list(coordinate_names)
        knee_indices = [
            coords.index(c) for c in ("knee_angle_r", "knee_angle_l") if c in coords
        ]
        if knee_indices:
            knee_deg = np.degrees(np.mean(q[:, knee_indices], axis=1))
            min_knee = float(np.min(knee_deg))
            # Standard squat standing phase requires extended knees (< 25 deg).
            # Deep squat bottom is >= 90 deg.
            if min_knee >= 45.0:
                # Lifter never approaches standing: report single honest bottom segment.
                label = f"bottom (engine window {t_start:.1f}-{t_end:.1f} s)"
                return [(label, t_start, t_end)]

    duration = t_end - t_start
    if lift in ("deadlift", "snatch", "clean_and_jerk"):
        # The partial 0.2 s engine window covers the pull only (knee pass),
        # not the full cycle (floor setup, lockout, catch, jerk, etc.).
        if duration <= 0.5:
            label = f"pull (engine window {t_start:.1f}-{t_end:.1f} s)"
            return [(label, t_start, t_end)]

    # Fallback to standard phase fractions only when motion covers full cycle
    phases = lift_phases(lift)
    segments = []
    for i, phase in enumerate(phases):
        t_phase_start = t_start + phase.t_frac * duration
        t_phase_end = (
            t_start + phases[i + 1].t_frac * duration if i + 1 < len(phases) else t_end
        )
        segments.append((phase.name, t_phase_start, t_phase_end))
    return segments


def _summarize_phases(
    times: np.ndarray,
    activations: np.ndarray,
    forces: np.ndarray,
    muscle_names: list[str],
    lift: str,
    coordinate_names: tuple[str, ...] | list[str] | None = None,
    q: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarize muscle group metrics per detected lift motion phase."""
    if coordinate_names is not None and q is not None:
        segments = detect_motion_phases(lift, times, coordinate_names, q)
    else:
        duration = float(times[-1]) - float(times[0])
        phases = lift_phases(lift)
        segments = [
            (
                p.name,
                float(times[0]) + p.t_frac * duration,
                (
                    float(times[0]) + phases[i + 1].t_frac * duration
                    if i + 1 < len(phases)
                    else float(times[-1])
                ),
            )
            for i, p in enumerate(phases)
        ]

    phase_summary: dict[str, dict[str, Any]] = {}
    for name, t_phase_start, t_phase_end in segments:
        mask = (times >= t_phase_start) & (times <= t_phase_end)
        if not np.any(mask):
            continue
        phase_act = activations[mask]
        phase_frc = forces[mask]
        phase_summary[name] = _summarize_groups(phase_act, phase_frc, muscle_names)

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
    except (RuntimeError, ValueError) as exc:  # OpenSim raises RuntimeError
        logger.warning("Moco pilot failed: %s", exc)
        return {
            "solver": "opensim.MocoInverse",
            "converged": False,
            "status": f"Failed: {exc}",
            "error": str(exc),
        }


def _run_lift_ik_id(
    lift: str,
    stride: int,
) -> tuple[LiftTargets, Any, ikid.IKResult, ikid.IDResult]:
    path = SAME_INPUT_LIFT_BUNDLES[lift]
    assert path is not None

    targets = lift_targets(lift, path, stride=stride)
    base_model = resolve_base_model()
    scales = ikid.bundle_segment_scales(targets, base_model)
    build, _reach = reaching_lift_build(lift, targets, base_model, scales)
    pelvis = ikid.pelvis_landmark_targets(build, targets, base_model)
    ik = ikid.run_inverse_kinematics(build, targets, pelvis)
    inverse = ikid.run_inverse_dynamics(build, targets, ik)
    return targets, build, ik, inverse


def _prepare_scratch_assets(
    lift: str,
    build_model: Any,
    ik: ikid.IKResult,
    targets: LiftTargets,
    scratch_dir: Path | None,
) -> tuple[Path, Path, Path]:
    if scratch_dir is not None:
        work_dir = scratch_dir
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix=f"opensim_lift_muscles_{lift}_"))

    actuated_model = _prepare_actuated_model(build_model)
    model_path = work_dir / f"{lift}_msk.osim"
    actuated_model.printToXML(str(model_path))

    coords_dict = {name: ik.q[:, j] for j, name in enumerate(ik.coordinates)}
    coords_path = work_dir / "coordinates.sto"
    write_sto(coords_path, targets.time_s, coords_dict)

    _, loads_xml = _build_external_loads(targets, actuated_model, work_dir)
    return model_path, coords_path, loads_xml


def _solve_lift_frames(
    basis: MuscleBasis,
    ik: ikid.IKResult,
    inverse: ikid.IDResult,
    leg_indices: list[int],
    time_s: np.ndarray,
    *,
    residual_gate_passed: bool = True,
) -> LiftSolveResults:
    n_frames = len(time_s)
    n_muscles = len(basis.muscles)
    n_leg_coords = len(leg_indices)

    activations = np.zeros((n_frames, n_muscles))
    forces = np.zeros((n_frames, n_muscles))
    reserves = np.zeros((n_frames, n_leg_coords))
    tau_muscles = np.zeros((n_frames, n_leg_coords))
    tau_targets = np.zeros((n_frames, n_leg_coords))
    cap_pos = np.zeros((n_frames, n_leg_coords))
    cap_neg = np.zeros((n_frames, n_leg_coords))
    passive_tau = np.zeros((n_frames, n_leg_coords))
    moment_balance_errors = np.zeros(n_frames)

    v_coords = np.gradient(ik.q, time_s, axis=0)

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

        cap_pos[i] = (np.clip(moment, 0, None) * active[:, None]).sum(axis=0)
        cap_neg[i] = (np.clip(moment, None, 0) * active[:, None]).sum(axis=0)
        passive_tau[i] = (moment * passive[:, None]).sum(axis=0)

        # Moment balance residual: tau_m + r == tau
        residual = tau_m + sol.reserve - tau_leg
        moment_balance_errors[i] = float(np.max(np.abs(residual)))

    frame_reserve_shares = [
        reserve_share(tau_targets[i], reserves[i]) for i in range(n_frames)
    ]
    mean_share = float(np.mean(frame_reserve_shares))
    peak_share = float(np.max(frame_reserve_shares))
    total_share = float(np.sum(np.abs(reserves)) / np.sum(np.abs(tau_targets)))
    classification = classify_lift_reserves(
        total_share, residual_gate_passed=residual_gate_passed
    )

    return LiftSolveResults(
        activations=activations,
        forces=forces,
        reserves=reserves,
        tau_muscles=tau_muscles,
        tau_targets=tau_targets,
        cap_pos=cap_pos,
        cap_neg=cap_neg,
        passive_tau=passive_tau,
        moment_balance_errors=moment_balance_errors,
        mean_share=mean_share,
        peak_share=peak_share,
        total_share=total_share,
        classification=classification,
    )


def evaluate_lift_muscles(
    lift: str,
    *,
    stride: int = 5,
    scratch_dir: Path | None = None,
    run_moco: bool = True,
) -> dict[str, Any]:
    """Compute muscle activations, forces, and reserve shares for *lift*.

    Generic across all barbell lifts. For lifts lacking an IK/ID bundle or
    supine posture, returns an honest ``unavailable`` record.
    """
    require(lift in LIFTS, f"unknown lift {lift!r}; choose from {list(LIFTS)}")

    if lift == "bench_press":
        return unavailable_lift_record(
            lift,
            "bench press IK does not pose a supine lifter; standing posture on bench is unphysical",
        )

    bundle_path = SAME_INPUT_LIFT_BUNDLES.get(lift)
    if bundle_path is None or not bundle_path.is_file():
        return unavailable_lift_record(
            lift,
            f"no LIFT-5 same-input bundle is registered for {lift!r} in "
            "SAME_INPUT_LIFT_BUNDLES (#11745), so there is no input to fit",
        )

    if not is_engine_available("opensim"):
        return unavailable_lift_record(
            lift,
            "Native OpenSim runtime is not available in current environment",
        )

    residual_status = load_ik_id_residual_status(lift)
    residual_gate_passed = (
        residual_status["forces_within"] and residual_status["moments_within"]
    )

    targets, build, ik, inverse = _run_lift_ik_id(lift, stride)
    model_path, coords_path, loads_xml = _prepare_scratch_assets(
        lift, build.model, ik, targets, scratch_dir
    )

    # Setup MuscleBasis with rigid-tendon Millard muscles
    _make_muscles_rigid_tendon(build.model)
    basis = MuscleBasis(build.model, ik.coordinates)
    leg_coords = leg_coordinates()
    leg_indices = [ik.coordinates.index(c) for c in leg_coords]

    solve = _solve_lift_frames(
        basis,
        ik,
        inverse,
        leg_indices,
        targets.time_s,
        residual_gate_passed=residual_gate_passed,
    )
    major_groups = _summarize_groups(
        solve.activations, solve.forces, basis.muscle_names
    )
    phases = _summarize_phases(
        targets.time_s,
        solve.activations,
        solve.forces,
        basis.muscle_names,
        lift,
        coordinate_names=ik.coordinates,
        q=ik.q,
    )
    diag = compute_squat_diagnostics(
        build.model, ik, leg_coords, targets, solve, major_groups, phases
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

    return format_squat_solution(
        lift,
        targets,
        basis.muscle_names,
        leg_coords,
        solve,
        diag,
        moco_record,
        RAJAGOPAL_MODEL_CITATION,
    )
