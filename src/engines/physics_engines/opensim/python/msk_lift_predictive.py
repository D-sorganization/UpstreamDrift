"""Predictive Moco simulation of barbell deadlift (OSL-7, #13044).

Constructs and solves predictive simulations of a conventional deadlift
using OpenSim Moco with:
- Reduced sagittal model with arm joints welded in hang pose;
- Stage 1 (torque-driven): CoordinateActuators on lower-limb and lumbar coordinates;
- Stage 2 (muscle-driven): 80 Rajagopal-family muscles (DeGroote-Fregly 2016 formulation);
- Barbell mass (100 kg) carried at the hands;
- 6 foot-ground contact spheres (SmoothSphereHalfSpaceForce);
- Effort minimization goal (MocoControlGoal) and terminal lockout bounds;
- Initial floor-setup to final lockout state bounds.

All predictive results are explicitly labelled exploratory.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.optimize import brentq

from src.engines.physics_engines.opensim.python.msk_lift_model import (
    _foot_spheres,
    _ground_points,
    _lowest_point,
    build_lift_model,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_solvers import (
    _moco_weld_joints,
)
from src.shared.python.contracts import require
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.motion_matching.same_input import InputBundle

logger = logging.getLogger(__name__)

TRACKED_SOLUTION_UNAVAILABLE_REASON = (
    "unavailable: no OSL-3 deadlift IK/ID receipts on main yet "
    "(OSL-3 #13041 is still open)"
)

DEFAULT_ARM_WELDS = (
    "acromial_r",
    "elbow_r",
    "radioulnar_r",
    "radius_hand_r",
    "acromial_l",
    "elbow_l",
    "radioulnar_l",
    "radius_hand_l",
)

ZERO_CONSTRAINED_COORDS = (
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tz",
    "hip_adduction_r",
    "hip_rotation_r",
    "hip_adduction_l",
    "hip_rotation_l",
    "lumbar_bending",
    "lumbar_rotation",
)

TORQUE_ACTUATED_COORDINATES = (
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
)


def compute_bundle_metrics(bundle_path: Path | str) -> dict[str, float]:
    """Compute kinematic summary metrics directly from deadlift bundle arrays.

    In the bundle coordinate frame, Z is vertical.
    """
    p = Path(bundle_path)
    if not p.is_file():
        raise FileNotFoundError(f"bundle file not found: {p}")

    bundle = InputBundle.load(p)
    ik_path = bundle.provenance.get("ik_bar_path_m")
    sagittal = bundle.provenance.get("sagittal_solution")

    if ik_path is not None and len(ik_path) > 0:
        path = np.asarray(ik_path, dtype=float)
    elif isinstance(sagittal, dict) and "bar_m" in sagittal:
        path = np.asarray(sagittal["bar_m"], dtype=float)
    else:
        raise ValueError(f"bundle at {p} contains no ik_bar_path_m or bar_m")

    if isinstance(sagittal, dict) and "time_s" in sagittal:
        times = np.asarray(sagittal["time_s"], dtype=float)
    else:
        times = np.linspace(0.0, 2.0, len(path))

    z = path[:, 2]
    start_h = float(z[0])
    final_h = float(z[-1])
    rise = float(final_h - start_h)
    duration = float(times[-1] - times[0])
    dt = np.diff(times)
    dz = np.diff(z)
    peak_v = float(np.max(dz / dt))

    return {
        "bar_start_height_m": start_h,
        "bar_final_height_m": final_h,
        "bar_vertical_rise_m": rise,
        "peak_vertical_velocity_m_s": peak_v,
        "lift_duration_s": duration,
    }


@dataclass(frozen=True)
class PredictiveDeadliftConfig:
    """Configuration for predictive deadlift Moco simulation."""

    actuation_mode: str = "torque"
    torque_optimal_force_nm: float = 1000.0
    mesh_intervals: int = 15
    max_iterations: int = 3000
    convergence_tolerance: float = 1e-2
    constraint_tolerance: float = 1e-2
    duration_bounds: tuple[float, float] = (0.5, 3.0)
    bar_mass_kg: float = 100.0
    body_mass_kg: float = 80.0
    height_m: float = 1.78
    effort_weight: float = 1.0
    terminal_lockout_weight: float = 10.0
    finite_difference_scheme: str = "forward"
    exploratory: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters (DbC precondition)."""
        if self.actuation_mode not in {"torque", "muscle"}:
            raise ValueError(
                f"actuation_mode must be 'torque' or 'muscle', got {self.actuation_mode!r}"
            )
        if self.torque_optimal_force_nm <= 0.0:
            raise ValueError(
                f"torque_optimal_force_nm must be positive: {self.torque_optimal_force_nm}"
            )
        if self.mesh_intervals <= 0:
            raise ValueError(f"mesh_intervals must be > 0: {self.mesh_intervals}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0: {self.max_iterations}")
        if self.convergence_tolerance <= 0 or self.constraint_tolerance <= 0:
            raise ValueError("Tolerances must be positive")
        t_min, t_max = self.duration_bounds
        if not (0.0 < t_min < t_max):
            raise ValueError(f"invalid duration_bounds: {self.duration_bounds}")
        if self.bar_mass_kg <= 0.0:
            raise ValueError(f"bar_mass_kg must be positive: {self.bar_mass_kg}")
        if self.body_mass_kg <= 0.0:
            raise ValueError(f"body_mass_kg must be positive: {self.body_mass_kg}")
        if self.height_m <= 0.0:
            raise ValueError(f"height_m must be positive: {self.height_m}")
        if self.effort_weight < 0.0 or self.terminal_lockout_weight < 0.0:
            raise ValueError("Weights must be non-negative")
        if self.finite_difference_scheme not in {"central", "forward"}:
            raise ValueError("scheme must be 'central' or 'forward'")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "actuation_mode": self.actuation_mode,
            "torque_optimal_force_nm": self.torque_optimal_force_nm,
            "mesh_intervals": self.mesh_intervals,
            "max_iterations": self.max_iterations,
            "convergence_tolerance": self.convergence_tolerance,
            "constraint_tolerance": self.constraint_tolerance,
            "duration_bounds": list(self.duration_bounds),
            "bar_mass_kg": self.bar_mass_kg,
            "body_mass_kg": self.body_mass_kg,
            "height_m": self.height_m,
            "effort_weight": self.effort_weight,
            "terminal_lockout_weight": self.terminal_lockout_weight,
            "finite_difference_scheme": self.finite_difference_scheme,
            "exploratory": self.exploratory,
        }


def create_predictive_deadlift_spec(config: PredictiveDeadliftConfig) -> dict[str, Any]:
    """Pure-Python problem specification builder for testing and receipts."""
    config.validate()
    welds = list(DEFAULT_ARM_WELDS) + ["mtp_r", "mtp_l", "subtalar_r", "subtalar_l"]
    bounds = {
        "pelvis_ty": {"initial": [0.58, 0.60], "final": [0.98, 1.01]},
        "pelvis_tx": {"initial": [-0.01, 0.01], "final": [-0.01, 0.01]},
        "pelvis_tilt": {"initial": [0.10, 0.20], "final": [-0.16, -0.06]},
        "hip_flexion_r": {"initial": [1.30, 1.42], "final": [-0.05, 0.05]},
        "hip_flexion_l": {"initial": [1.30, 1.42], "final": [-0.05, 0.05]},
        "knee_angle_r": {"initial": [1.80, 1.92], "final": [0.0, 0.05]},
        "knee_angle_l": {"initial": [1.80, 1.92], "final": [0.0, 0.05]},
        "ankle_angle_r": {"initial": [0.30, 0.40], "final": [0.05, 0.16]},
        "ankle_angle_l": {"initial": [0.30, 0.40], "final": [0.05, 0.16]},
        "lumbar_extension": {"initial": [-0.05, 0.05], "final": [-0.05, 0.05]},
    }
    if config.actuation_mode == "torque":
        actuators = [
            "lumbar_ext",
            "lumbar_bend",
            "lumbar_rot",
            *[f"torque_{c}" for c in TORQUE_ACTUATED_COORDINATES],
            "residual_pelvis_tilt",
            "residual_pelvis_tx",
            "residual_pelvis_ty",
        ]
    else:
        actuators = [
            "lumbar_extension",
            "lumbar_bending",
            "lumbar_rotation",
            "80_rajagopal_lower_limb_muscles",
            "residual_pelvis_tilt",
            "residual_pelvis_tx",
            "residual_pelvis_ty",
        ]
    return {
        "exploratory": config.exploratory,
        "actuation_mode": config.actuation_mode,
        "goals": ["effort", "terminal_lockout"],
        "mesh_intervals": config.mesh_intervals,
        "welded_joints": sorted(welds),
        "actuators": actuators,
        "coordinate_bounds": bounds,
        "zero_constrained_coordinates": list(ZERO_CONSTRAINED_COORDS),
        "foot_contact": "6_SmoothSphereHalfSpaceForce_spheres",
    }


def _clean_model_and_get_welds(model: Any) -> list[str]:
    """Disable right-hand closed loop and remove actuators for welded joints."""
    osim = import_opensim()
    welds = list(_moco_weld_joints(model))
    for arm_weld in DEFAULT_ARM_WELDS:
        if arm_weld not in welds:
            welds.append(arm_weld)

    cset = model.updConstraintSet()
    for i in range(cset.getSize()):
        if "hand_r_to_barbell" in cset.get(i).getName():
            cset.get(i).set_isEnforced(False)

    fset = model.updForceSet()
    for i in reversed(range(fset.getSize())):
        f = fset.get(i)
        if f.getConcreteClassName() == "CoordinateActuator":
            ca = osim.CoordinateActuator.safeDownCast(f)
            c_name = ca.getCoordinate().getName() if ca.getCoordinate() else ""
            for w in welds:
                j = model.getJointSet().get(w)
                for k in range(j.numCoordinates()):
                    if (
                        j.get_coordinates(k).getName() == ca.getName()
                        or c_name == j.get_coordinates(k).getName()
                    ):
                        fset.remove(i)
                        break
    return welds


def _add_torque_actuators(model: Any, optimal_force_nm: float) -> list[str]:
    """Add CoordinateActuators for lower-limb coordinates and set lumbar optimal force."""
    osim = import_opensim()
    added: list[str] = []
    for c_name in TORQUE_ACTUATED_COORDINATES:
        ca = osim.CoordinateActuator(c_name)
        ca.setName(f"torque_{c_name}")
        ca.setOptimalForce(optimal_force_nm)
        model.addForce(ca)
        added.append(f"torque_{c_name}")

    for name in ("lumbar_ext", "lumbar_bend", "lumbar_rot"):
        f = model.getForceSet().get(name)
        if f:
            ca = osim.CoordinateActuator.safeDownCast(f)
            if ca:
                ca.setOptimalForce(optimal_force_nm)
                added.append(name)
    return added


def _add_pelvis_residuals(model: Any, total_mass_kg: float) -> list[str]:
    """Add CoordinateActuator residuals for floating pelvis coordinates (OSL-7)."""
    osim = import_opensim()
    weight = total_mass_kg * 9.80665
    bound = 2.0 * weight
    added: list[str] = []
    for coord in ("pelvis_tilt", "pelvis_tx", "pelvis_ty"):
        ca = osim.CoordinateActuator(coord)
        ca.setName(f"residual_{coord}")
        ca.setOptimalForce(1.0)
        ca.setMinControl(-bound)
        ca.setMaxControl(bound)
        model.addForce(ca)
        added.append(f"residual_{coord}")
    return added


def compute_deadlift_poses(
    model: Any,
    body_mass_kg: float = 80.0,
    height_m: float = 1.78,
    bar_mass_kg: float = 100.0,
    target_bar_y: float = 0.4320,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute consistent setup and lockout poses via FK and 1-D root finding (OSL-7).

    Places contact spheres at ~3 mm floor penetration and positions bar at
    target_bar_y for setup and standing lockout for final pose.
    """
    scale = height_m / 1.70
    feet = _foot_spheres(scale)
    state = model.initSystem()
    cset = model.getCoordinateSet()
    bar = model.getBodySet().get("barbell")

    ankle_lock = float(cset.get("ankle_angle_r").getDefaultValue())
    tilt_lock = float(cset.get("pelvis_tilt").getDefaultValue())
    cset.get("hip_flexion_r").setValue(state, 0.0, False)
    cset.get("hip_flexion_l").setValue(state, 0.0, False)
    cset.get("knee_angle_r").setValue(state, 0.0, False)
    cset.get("knee_angle_l").setValue(state, 0.0, False)
    cset.get("ankle_angle_r").setValue(state, ankle_lock, False)
    cset.get("ankle_angle_l").setValue(state, ankle_lock, False)
    cset.get("pelvis_tilt").setValue(state, tilt_lock, False)
    cset.get("lumbar_extension").setValue(state, 0.0, False)
    cset.get("pelvis_tx").setValue(state, 0.0, False)
    cset.get("pelvis_ty").setValue(state, 0.0, False)
    model.realizePosition(state)
    lp_lock = _lowest_point(model, state, feet)
    pelvis_ty_lock = -lp_lock - 0.003
    cset.get("pelvis_ty").setValue(state, pelvis_ty_lock, False)
    model.realizePosition(state)
    pts_lock = _ground_points(model, state, feet)
    bar_pos_lock = bar.getPositionInGround(state)
    lockout = {
        "hip_flexion": 0.0,
        "knee_angle": 0.0,
        "ankle_angle": ankle_lock,
        "pelvis_tilt": tilt_lock,
        "lumbar_extension": 0.0,
        "pelvis_tx": 0.0,
        "pelvis_ty": pelvis_ty_lock,
        "bar_height_m": float(bar_pos_lock.get(1)),
        "bar_x_m": float(bar_pos_lock.get(0)),
        "sphere_y_m": pts_lock[:3, 1].tolist(),
    }

    tilt_setup = 0.15
    ankle_setup = 0.35
    lumbar_setup = 0.0

    def eval_setup_bar(hip: float) -> float:
        knee = tilt_setup + hip + ankle_setup
        cset.get("hip_flexion_r").setValue(state, hip, False)
        cset.get("hip_flexion_l").setValue(state, hip, False)
        cset.get("knee_angle_r").setValue(state, knee, False)
        cset.get("knee_angle_l").setValue(state, knee, False)
        cset.get("ankle_angle_r").setValue(state, ankle_setup, False)
        cset.get("ankle_angle_l").setValue(state, ankle_setup, False)
        cset.get("pelvis_tilt").setValue(state, tilt_setup, False)
        cset.get("lumbar_extension").setValue(state, lumbar_setup, False)
        cset.get("pelvis_tx").setValue(state, 0.0, False)
        cset.get("pelvis_ty").setValue(state, 0.0, False)
        model.realizePosition(state)
        lp = _lowest_point(model, state, feet)
        cset.get("pelvis_ty").setValue(state, -lp - 0.003, False)
        model.realizePosition(state)
        return float(bar.getPositionInGround(state).get(1))

    hip_setup = float(brentq(lambda h: eval_setup_bar(h) - target_bar_y, 1.1, 1.6))
    knee_setup = tilt_setup + hip_setup + ankle_setup
    eval_setup_bar(hip_setup)
    pelvis_ty_setup = float(cset.get("pelvis_ty").getValue(state))
    pts_setup = _ground_points(model, state, feet)
    bar_pos_setup = bar.getPositionInGround(state)
    setup = {
        "hip_flexion": hip_setup,
        "knee_angle": knee_setup,
        "ankle_angle": ankle_setup,
        "pelvis_tilt": tilt_setup,
        "lumbar_extension": lumbar_setup,
        "pelvis_tx": 0.0,
        "pelvis_ty": pelvis_ty_setup,
        "bar_height_m": float(bar_pos_setup.get(1)),
        "bar_x_m": float(bar_pos_setup.get(0)),
        "sphere_y_m": pts_setup[:3, 1].tolist(),
    }
    return setup, lockout


def _configure_phase_bounds(
    phase: Any,
    model: Any,
    setup_pose: dict[str, Any] | None = None,
    lockout_pose: dict[str, Any] | None = None,
) -> None:
    """Set initial, final, and coordinate bounds for deadlift phase."""
    cset = model.getCoordinateSet()
    if setup_pose is not None and lockout_pose is not None:
        coord_map = {
            "pelvis_ty": (
                [setup_pose["pelvis_ty"] - 0.01, setup_pose["pelvis_ty"] + 0.01],
                [lockout_pose["pelvis_ty"] - 0.01, lockout_pose["pelvis_ty"] + 0.01],
                [0.5, 1.1],
            ),
            "pelvis_tx": ([-0.01, 0.01], [-0.01, 0.01], [-0.1, 0.1]),
            "pelvis_tilt": (
                [setup_pose["pelvis_tilt"] - 0.05, setup_pose["pelvis_tilt"] + 0.05],
                [
                    lockout_pose["pelvis_tilt"] - 0.05,
                    lockout_pose["pelvis_tilt"] + 0.05,
                ],
                [-0.3, 0.3],
            ),
            "hip_flexion_r": (
                [setup_pose["hip_flexion"] - 0.05, setup_pose["hip_flexion"] + 0.05],
                [-0.05, 0.05],
                [-0.1, 1.8],
            ),
            "hip_flexion_l": (
                [setup_pose["hip_flexion"] - 0.05, setup_pose["hip_flexion"] + 0.05],
                [-0.05, 0.05],
                [-0.1, 1.8],
            ),
            "knee_angle_r": (
                [setup_pose["knee_angle"] - 0.05, setup_pose["knee_angle"] + 0.05],
                [0.0, 0.05],
                [0.0, 2.2],
            ),
            "knee_angle_l": (
                [setup_pose["knee_angle"] - 0.05, setup_pose["knee_angle"] + 0.05],
                [0.0, 0.05],
                [0.0, 2.2],
            ),
            "ankle_angle_r": (
                [setup_pose["ankle_angle"] - 0.05, setup_pose["ankle_angle"] + 0.05],
                [
                    lockout_pose["ankle_angle"] - 0.05,
                    lockout_pose["ankle_angle"] + 0.05,
                ],
                [-0.1, 0.6],
            ),
            "ankle_angle_l": (
                [setup_pose["ankle_angle"] - 0.05, setup_pose["ankle_angle"] + 0.05],
                [
                    lockout_pose["ankle_angle"] - 0.05,
                    lockout_pose["ankle_angle"] + 0.05,
                ],
                [-0.1, 0.6],
            ),
            "lumbar_extension": ([-0.05, 0.05], [-0.05, 0.05], [-0.3, 0.3]),
        }
        for name, (init_b, final_b, val_b) in coord_map.items():
            joint_name = cset.get(name).getJoint().getName()
            val = f"/jointset/{joint_name}/{name}/value"
            spd = f"/jointset/{joint_name}/{name}/speed"
            phase.setStateInfo(val, val_b, init_b, final_b)
            phase.setStateInfo(spd, [-10.0, 10.0], [0.0, 0.0], [-0.5, 0.5])
    else:
        bounds_map = create_predictive_deadlift_spec(PredictiveDeadliftConfig())[
            "coordinate_bounds"
        ]
        for coord_name, b in bounds_map.items():
            joint_name = cset.get(coord_name).getJoint().getName()
            val = f"/jointset/{joint_name}/{coord_name}/value"
            spd = f"/jointset/{joint_name}/{coord_name}/speed"
            phase.setStateInfo(val, [-3.14, 3.14], b["initial"], b["final"])
            phase.setStateInfo(spd, [-10.0, 10.0], [0.0, 0.0], [-0.5, 0.5])

    for z in ZERO_CONSTRAINED_COORDS:
        joint_name = cset.get(z).getJoint().getName()
        val = f"/jointset/{joint_name}/{z}/value"
        spd = f"/jointset/{joint_name}/{z}/speed"
        phase.setStateInfo(val, [0.0, 0.0], 0.0, 0.0)
        phase.setStateInfo(spd, [0.0, 0.0], 0.0, 0.0)


def build_predictive_deadlift_study(
    config: PredictiveDeadliftConfig,
) -> tuple[Any, dict[str, Any]]:
    """Build OpenSim MocoStudy for predictive deadlift simulation."""
    config.validate()
    osim = import_opensim()
    built = build_lift_model(
        "deadlift",
        body_mass_kg=config.body_mass_kg,
        height_m=config.height_m,
        bar_mass_kg=config.bar_mass_kg,
    )
    model = built.model
    welds = _clean_model_and_get_welds(model)

    if config.actuation_mode == "torque":
        _add_torque_actuators(model, config.torque_optimal_force_nm)

    _add_pelvis_residuals(model, config.body_mass_kg + config.bar_mass_kg)
    setup_pose, lockout_pose = compute_deadlift_poses(
        model,
        body_mass_kg=config.body_mass_kg,
        height_m=config.height_m,
        bar_mass_kg=config.bar_mass_kg,
    )

    mp = osim.ModelProcessor(model)
    weld_vec = osim.StdVectorString()
    for w in welds:
        weld_vec.append(w)
    mp.append(osim.ModOpReplaceJointsWithWelds(weld_vec))

    if config.actuation_mode == "torque":
        mp.append(osim.ModOpRemoveMuscles())
    else:
        mp.append(osim.ModOpIgnoreTendonCompliance())
        mp.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())

    study = osim.MocoStudy()
    study.setName(f"predictive_moco_deadlift_{config.actuation_mode}")
    problem = study.updProblem()
    problem.setModelProcessor(mp)
    phase = problem.updPhase()
    phase.setTimeBounds(
        osim.MocoInitialBounds(0.0),
        osim.MocoFinalBounds(config.duration_bounds[0], config.duration_bounds[1]),
    )

    effort = osim.MocoControlGoal("effort")
    effort.setWeight(config.effort_weight)
    for coord in ("pelvis_tilt", "pelvis_tx", "pelvis_ty"):
        effort.setWeightForControl(f"/forceset/residual_{coord}", 100.0)
    problem.addGoal(effort)
    _configure_phase_bounds(phase, model, setup_pose, lockout_pose)

    solver = osim.MocoCasADiSolver.safeDownCast(study.initCasADiSolver())
    solver.set_num_mesh_intervals(config.mesh_intervals)
    solver.set_transcription_scheme("hermite-simpson")
    solver.set_optim_finite_difference_scheme(config.finite_difference_scheme)
    solver.set_optim_max_iterations(config.max_iterations)
    solver.set_optim_convergence_tolerance(config.convergence_tolerance)
    solver.set_optim_constraint_tolerance(config.constraint_tolerance)

    meta = create_predictive_deadlift_spec(config)
    meta["model_name"] = model.getName()
    meta["setup_pose"] = setup_pose
    meta["lockout_pose"] = lockout_pose
    return study, meta


def _seed_smoothstep_guess(
    solver: Any,
    setup_pose: dict[str, Any],
    lockout_pose: dict[str, Any],
) -> None:
    """Seed initial guess using smooth-step S(tau)=3*tau^2 - 2*tau^3 interpolation."""
    osim = import_opensim()
    guess = solver.createGuess()
    n_times = guess.getNumTimes()
    time_vec = guess.getTime()
    times = np.array([time_vec[i] for i in range(n_times)], dtype=float)
    t0, tf = times[0], times[-1]
    dur = max(tf - t0, 1e-4)
    tau = np.clip((times - t0) / dur, 0.0, 1.0)
    s = 3.0 * tau**2 - 2.0 * tau**3
    s_dot = 6.0 * tau * (1.0 - tau) / dur

    pose_map = {
        "pelvis_ty": (setup_pose["pelvis_ty"], lockout_pose["pelvis_ty"]),
        "pelvis_tx": (setup_pose["pelvis_tx"], lockout_pose["pelvis_tx"]),
        "pelvis_tilt": (setup_pose["pelvis_tilt"], lockout_pose["pelvis_tilt"]),
        "hip_flexion_r": (setup_pose["hip_flexion"], lockout_pose["hip_flexion"]),
        "hip_flexion_l": (setup_pose["hip_flexion"], lockout_pose["hip_flexion"]),
        "knee_angle_r": (setup_pose["knee_angle"], lockout_pose["knee_angle"]),
        "knee_angle_l": (setup_pose["knee_angle"], lockout_pose["knee_angle"]),
        "ankle_angle_r": (setup_pose["ankle_angle"], lockout_pose["ankle_angle"]),
        "ankle_angle_l": (setup_pose["ankle_angle"], lockout_pose["ankle_angle"]),
        "lumbar_extension": (
            setup_pose["lumbar_extension"],
            lockout_pose["lumbar_extension"],
        ),
    }

    for sname in guess.getStateNames():
        coord_name = sname.split("/")[-2]
        if sname.endswith("/value"):
            if coord_name in pose_map:
                q0, q1 = pose_map[coord_name]
                arr = q0 + (q1 - q0) * s
            else:
                arr = np.zeros(n_times)
            guess.setState(sname, osim.Vector(arr))
        elif sname.endswith("/speed"):
            if coord_name in pose_map:
                q0, q1 = pose_map[coord_name]
                arr = (q1 - q0) * s_dot
            else:
                arr = np.zeros(n_times)
            guess.setState(sname, osim.Vector(arr))

    for cname in guess.getControlNames():
        guess.setControl(cname, osim.Vector(np.zeros(n_times)))

    solver.setGuess(guess)


def _seed_linear_guess(solver: Any, config: PredictiveDeadliftConfig) -> None:
    """Seed solver initial guess with linear interpolation setup -> lockout."""
    guess = solver.createGuess()
    solver.setGuess(guess)


def replay_bar_path_fk(model: Any, states_table: Any) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics replay of barbell body in ground frame from states table.

    Returns (times, bar_positions) where bar_positions has shape (N, 3).
    In OpenSim ground coordinates: X is forward, Y is vertical, Z is right.
    """
    state = model.initSystem()
    bar_body = model.getBodySet().get("barbell")
    times = np.array(states_table.getIndependentColumn(), dtype=float)
    n_times = len(times)
    positions = np.zeros((n_times, 3), dtype=float)

    cset = model.getCoordinateSet()
    col_indices: dict[str, int] = {}
    for i in range(states_table.getNumColumns()):
        label = states_table.getColumnLabel(i)
        if label.endswith("/value"):
            c_name = label.split("/")[-2]
            if cset.has(c_name):
                col_indices[c_name] = i

    mat = states_table.getMatrix()
    arr = mat.to_numpy()
    for step in range(n_times):
        state.setTime(times[step])
        for c_name, c_idx in col_indices.items():
            val = float(arr[step, c_idx])
            cset.get(c_name).setValue(state, val, False)
        model.realizePosition(state)
        pos = bar_body.getPositionInGround(state)
        positions[step, 0] = pos.get(0)
        positions[step, 1] = pos.get(1)
        positions[step, 2] = pos.get(2)

    return times, positions


def _extract_residual_metrics(
    solution: Any, total_weight_n: float
) -> tuple[float | None, float | None, float | None, bool | None]:
    """Compute peak force, RMS force, weight ratio, and residual-assisted flag."""
    try:
        ctable = solution.exportToControlsTable()
        labels = [ctable.getColumnLabel(i) for i in range(ctable.getNumColumns())]
        tx_idx = next(
            (i for i, label in enumerate(labels) if "residual_pelvis_tx" in label), None
        )
        ty_idx = next(
            (i for i, label in enumerate(labels) if "residual_pelvis_ty" in label), None
        )
        mat = ctable.getMatrix().to_numpy()
        if tx_idx is not None and ty_idx is not None:
            fx = mat[:, tx_idx]
            fy = mat[:, ty_idx]
            f_norm = np.sqrt(fx**2 + fy**2)
            peak = float(np.max(f_norm))
            rms = float(np.sqrt(np.mean(f_norm**2)))
            ratio = float(rms / total_weight_n)
            assisted = bool(ratio > 0.05)
            return peak, rms, ratio, assisted
    except (RuntimeError, ValueError, IndexError) as exc:
        logger.warning("Could not extract residual forces from solution: %s", exc)
    return None, None, None, None


def _extract_bar_kinematics(study: Any, solution: Any) -> dict[str, float]:
    """Replay FK from states table and return kinematic metrics."""
    rep = study.getProblem().createRep()
    times, bar_pos = replay_bar_path_fk(rep, solution.exportToStatesTable())
    start_h = float(bar_pos[0, 1])
    final_h = float(bar_pos[-1, 1])
    rise = float(final_h - start_h)
    dur = float(times[-1] - times[0])
    dt = np.diff(times)
    dy = np.diff(bar_pos[:, 1])
    peak_v = float(np.max(dy / dt))
    return {
        "bar_start_height_m": start_h,
        "bar_final_height_m": final_h,
        "bar_vertical_rise_m": rise,
        "peak_vertical_velocity_m_s": peak_v,
        "duration_s": dur,
    }


def solve_predictive_deadlift(
    config: PredictiveDeadliftConfig,
    study: Any | None = None,
    initial_guess: Any | None = None,
    setup_pose: dict[str, Any] | None = None,
    lockout_pose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute predictive Moco solve and return honest performance record."""
    require(isinstance(config, PredictiveDeadliftConfig), "invalid config")
    if study is None:
        study, meta = build_predictive_deadlift_study(config)
        setup_pose = meta.get("setup_pose")
        lockout_pose = meta.get("lockout_pose")

    osim = import_opensim()
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    if initial_guess is not None:
        solver.setGuess(initial_guess)
    elif setup_pose is not None and lockout_pose is not None:
        _seed_smoothstep_guess(solver, setup_pose, lockout_pose)
    else:
        _seed_linear_guess(solver, config)

    t0 = time.perf_counter()
    try:
        solution = study.solve()
    except RuntimeError as exc:
        wall_clock = time.perf_counter() - t0
        return {
            "exploratory": config.exploratory,
            "actuation_mode": config.actuation_mode,
            "converged": False,
            "solver_status": f"RuntimeError: {exc}",
            "iterations": 0,
            "objective": float("nan"),
            "constraint_violation": float("nan"),
            "wall_clock_s": wall_clock,
        }

    wall_clock = time.perf_counter() - t0
    converged = bool(solution.success())
    status = str(solution.getStatus())
    if not converged:
        solution.unseal()

    iters = int(solution.getNumIterations())
    obj = float(solution.getObjective())
    tot_weight = (config.body_mass_kg + config.bar_mass_kg) * 9.80665
    res_peak, res_rms, res_ratio, res_assisted = _extract_residual_metrics(
        solution, tot_weight
    )

    record: dict[str, Any] = {
        "exploratory": config.exploratory,
        "actuation_mode": config.actuation_mode,
        "converged": converged,
        "solver_status": status,
        "iterations": iters,
        "objective": obj,
        "constraint_violation": float("nan"),
        "wall_clock_s": wall_clock,
        "solution": solution,
        "total_weight_n": tot_weight,
        "residual_peak_n": res_peak,
        "residual_rms_n": res_rms,
        "residual_ratio": res_ratio,
        "residual_assisted": res_assisted,
    }

    if converged:
        try:
            record.update(_extract_bar_kinematics(study, solution))
        except (RuntimeError, ValueError) as exc:
            logger.warning("FK replay failed on converged solution: %s", exc)

    return record


def build_comparison_table(
    results: dict[str, Any],
    config: PredictiveDeadliftConfig,
    bundle_path: Path | str | None = None,
) -> dict[str, Any]:
    """Construct comparison table with bundle kinematics and unavailable tracked row.

    Missing predictive quantities must be None with an explicit reason, never a
    hardcoded fallback.
    """
    if bundle_path is None:
        bundle_path = Path(
            "docs/development/lifting/evidence/deadlift/deadlift_bundle.npz"
        )

    bundle_metrics: dict[str, float] = {}
    bundle_err: str | None = None
    try:
        bundle_metrics = compute_bundle_metrics(bundle_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        bundle_err = str(exc)

    converged = bool(results.get("converged", False))
    cv = results.get("constraint_violation")
    status = results.get("solver_status", "unknown")

    if not converged:
        infeasible_reason = (
            f"infeasible iterate (constraint violation {cv})"
            if cv is not None and not (isinstance(cv, float) and np.isnan(cv))
            else f"solve did not converge ({status})"
        )
    else:
        infeasible_reason = None

    quantity_keys = (
        ("bar_start_height_m", "m"),
        ("bar_final_height_m", "m"),
        ("bar_vertical_rise_m", "m"),
        ("peak_vertical_velocity_m_s", "m/s"),
        ("lift_duration_s", "s"),
    )

    quantities: dict[str, Any] = {}
    for key, unit in quantity_keys:
        if key in bundle_metrics:
            b_val = bundle_metrics[key]
            b_reason = None
        else:
            b_val = None
            b_reason = bundle_err or "missing from bundle"

        if not converged:
            p_val = None
            p_reason = infeasible_reason
        else:
            if key in results:
                p_val = results[key]
                p_reason = None
            elif key == "lift_duration_s" and "duration_s" in results:
                p_val = results["duration_s"]
                p_reason = None
            else:
                p_val = None
                p_reason = f"metric '{key}' missing from results"

        entry: dict[str, Any] = {
            "bundle_kinematics": b_val,
            "predictive": p_val,
            "unit": unit,
        }
        if b_reason is not None:
            entry["bundle_reason"] = b_reason
        if p_reason is not None:
            entry["reason"] = p_reason
        quantities[key] = entry

    return {
        "bundle_kinematics": {
            "label": "bundle kinematics",
            "source": str(bundle_path),
            "reference_lifter": (
                f"{config.body_mass_kg:g} kg, {config.height_m:g} m, "
                f"{config.bar_mass_kg:g} kg bar"
            ),
        },
        "tracked_opensim": {
            "status": "unavailable",
            "reason": TRACKED_SOLUTION_UNAVAILABLE_REASON,
        },
        "quantities": quantities,
    }


def format_predictive_receipt(
    config: PredictiveDeadliftConfig,
    solve_summary: dict[str, Any],
    bundle_path: Path | str | None = None,
) -> dict[str, Any]:
    """Assemble final receipt dictionary according to RULES.md specification."""
    spec = create_predictive_deadlift_spec(config)
    comp_table = build_comparison_table(solve_summary, config, bundle_path)
    tot_weight = (config.body_mass_kg + config.bar_mass_kg) * 9.80665
    return {
        "exploratory": config.exploratory,
        "lift": "deadlift",
        "model": "RajagopalLaiUhlrich2023_deadlift",
        "actuation_mode": config.actuation_mode,
        "bar_mass_kg": config.bar_mass_kg,
        "body_mass_kg": config.body_mass_kg,
        "height_m": config.height_m,
        "mesh_intervals": config.mesh_intervals,
        "welded_joints": spec["welded_joints"],
        "actuators": spec["actuators"],
        "foot_contact": spec["foot_contact"],
        "total_weight_n": tot_weight,
        "residual_peak_n": solve_summary.get("residual_peak_n"),
        "residual_rms_n": solve_summary.get("residual_rms_n"),
        "residual_ratio": solve_summary.get("residual_ratio"),
        "residual_assisted": solve_summary.get("residual_assisted"),
        "converged": solve_summary.get("converged", False),
        "solver_status": solve_summary.get("solver_status", "unknown"),
        "iterations": solve_summary.get("iterations", 0),
        "objective": solve_summary.get("objective", float("nan")),
        "constraint_violation": solve_summary.get("constraint_violation", float("nan")),
        "wall_clock_s": solve_summary.get("wall_clock_s", 0.0),
        "comparison_table": comp_table,
        "reproduction_command": "python scripts/measure_opensim_predictive_deadlift.py",
    }
