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
from scipy.optimize import least_squares

from src.engines.physics_engines.opensim.python.msk_lift_model import (
    _foot_spheres,
    _ground_points,
    build_lift_model,
)
from src.engines.physics_engines.opensim.python.msk_lift_predictive_report import (
    _extract_bar_kinematics,
    _extract_residual_metrics,
    build_comparison_table,
    bundle_pose_targets,
    compute_bundle_metrics,
    replay_bar_path_fk,
    residual_moment_gate,
    residual_moment_metrics,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_solvers import (
    _moco_weld_joints,
)
from src.shared.python.contracts import require
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.motion_matching.same_input import InputBundle

logger = logging.getLogger(__name__)

#: Straight arms: elbows, forearms and wrists welded; the shoulders stay free
#: so the arms can hang vertically while the trunk inclines (OSL-7, #13044).
DEFAULT_ARM_WELDS = (
    "elbow_r",
    "radioulnar_r",
    "radius_hand_r",
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
    "arm_add_r",
    "arm_rot_r",
    "arm_add_l",
    "arm_rot_l",
)

#: The base model's shoulder CoordinateActuators (10 N m optimal force),
#: raised to the joint-torque scale once the shoulders are free.
SHOULDER_ACTUATORS = (
    "shoulder_flex_r",
    "shoulder_add_r",
    "shoulder_rot_r",
    "shoulder_flex_l",
    "shoulder_add_l",
    "shoulder_rot_l",
)

#: Every floating-base coordinate gets a residual.  The out-of-plane ones
#: (list, rotation, tz) are held at zero by the phase bounds, so without an
#: actuator their equations of motion can only balance if every lateral load
#: (the one-handed bar weld, asymmetric contact) cancels exactly, which made
#: stage 1 dynamically infeasible (OSL-7, #13044).
PELVIS_RESIDUAL_COORDINATES = (
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
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
    residual_force_scale: float = 2.0
    residual_effort_weight: float = 10.0
    planted_feet_weight: float = 1.0e4
    multibody_dynamics_mode: str = "explicit"
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
        if not (0.0 < t_min <= t_max):
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
        if self.multibody_dynamics_mode not in {"explicit", "implicit"}:
            raise ValueError("multibody_dynamics_mode must be 'explicit' or 'implicit'")
        if self.residual_force_scale <= 0.0 or self.residual_effort_weight < 0.0:
            raise ValueError(
                "residual_force_scale must be positive and residual_effort_weight "
                "non-negative"
            )
        if self.planted_feet_weight < 0.0:
            raise ValueError(
                f"planted_feet_weight must be non-negative: {self.planted_feet_weight}"
            )

    def residual_optimal_force_n(self) -> float:
        """Residual optimal force: ``residual_force_scale`` times total weight."""
        total = self.body_mass_kg + self.bar_mass_kg
        return self.residual_force_scale * total * 9.80665

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
            "planted_feet_weight": self.planted_feet_weight,
            "finite_difference_scheme": self.finite_difference_scheme,
            "exploratory": self.exploratory,
        }


def create_predictive_deadlift_spec(config: PredictiveDeadliftConfig) -> dict[str, Any]:
    """Pure-Python problem specification builder for testing and receipts."""
    config.validate()
    welds = list(DEFAULT_ARM_WELDS) + ["mtp_r", "mtp_l", "subtalar_r", "subtalar_l"]
    bounds = (
        "initial and final values within POSE_BOUND of the setup and lockout "
        "poses solved by compute_deadlift_poses"
    )
    if config.actuation_mode == "torque":
        actuators = [
            "lumbar_ext",
            "lumbar_bend",
            "lumbar_rot",
            *SHOULDER_ACTUATORS,
            *[f"torque_{c}" for c in TORQUE_ACTUATED_COORDINATES],
            *[f"residual_{c}" for c in PELVIS_RESIDUAL_COORDINATES],
        ]
    else:
        actuators = [
            "lumbar_extension",
            "lumbar_bending",
            "lumbar_rotation",
            *SHOULDER_ACTUATORS,
            "80_rajagopal_lower_limb_muscles",
            *[f"residual_{c}" for c in PELVIS_RESIDUAL_COORDINATES],
        ]
    return {
        "exploratory": config.exploratory,
        "actuation_mode": config.actuation_mode,
        "goals": ["effort", "planted_feet"],
        "planted_feet_weight": config.planted_feet_weight,
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

    added += set_actuator_optimal_force(
        model, ("lumbar_ext", "lumbar_bend", "lumbar_rot"), optimal_force_nm
    )
    return added


def set_actuator_optimal_force(
    model: Any, names: tuple[str, ...], optimal_force_nm: float
) -> list[str]:
    """Set the optimal force of the model's named CoordinateActuators."""
    require(
        optimal_force_nm > 0.0, f"optimal force must be positive: {optimal_force_nm}"
    )
    osim = import_opensim()
    for name in names:
        ca = osim.CoordinateActuator.safeDownCast(model.getForceSet().get(name))
        require(ca is not None, f"model has no CoordinateActuator {name!r}")
        ca.setOptimalForce(optimal_force_nm)
    return list(names)


def _add_pelvis_residuals(model: Any, optimal_force_n: float) -> list[str]:
    """Add CoordinateActuator residuals for floating pelvis coordinates (OSL-7).

    Controls are nondimensional in [-1, 1]; the optimal force carries the
    scale, so the effort goal sees O(1) residual controls like the joint
    torques instead of raw newtons.
    """
    if optimal_force_n <= 0.0:
        raise ValueError(f"optimal_force_n must be positive: {optimal_force_n}")
    osim = import_opensim()
    added: list[str] = []
    for coord in PELVIS_RESIDUAL_COORDINATES:
        ca = osim.CoordinateActuator(coord)
        ca.setName(f"residual_{coord}")
        ca.setOptimalForce(optimal_force_n)
        ca.setMinControl(-1.0)
        ca.setMaxControl(1.0)
        model.addForce(ca)
        added.append(f"residual_{coord}")
    return added


#: Contact-sphere floor penetration of the seated poses (m).
POSE_FLOOR_PENETRATION_M = 0.003
#: Largest equation residual a solved pose may keep (m).
POSE_SOLVE_TOLERANCE_M = 1e-6
PLANTED_FEET_FRAMES = ("/bodyset/calcn_r", "/bodyset/calcn_l")
PLANTED_FEET_SAMPLES = 11
PLANTED_FEET_TIME_MARGIN_S = 0.1

_LEG_POSE_KEYS = ("hip_flexion", "knee_angle", "ankle_angle")
_TRUNK_POSE_KEYS = ("pelvis_tilt", "lumbar_extension", "pelvis_tx", "pelvis_ty")


def _apply_pose(model: Any, state: Any, pose: dict[str, float]) -> None:
    """Set a bilateral sagittal pose on ``state`` and realize positions."""
    cset = model.getCoordinateSet()
    for key in _LEG_POSE_KEYS:
        for side in "rl":
            cset.get(f"{key}_{side}").setValue(state, pose[key], False)
    for side in "rl":
        cset.get(f"arm_flex_{side}").setValue(state, pose["arm_flex"], False)
    for key in _TRUNK_POSE_KEYS:
        cset.get(key).setValue(state, pose[key], False)
    model.realizePosition(state)


def _pose_features(model: Any, state: Any, feet: list[Any]) -> dict[str, Any]:
    """Translation-invariant geometry of the realized pose (x forward, y up)."""
    bodies = model.getBodySet()
    points = _ground_points(model, state, feet)
    right = points[:3]
    bar = bodies.get("barbell").getPositionInGround(state)
    hip = bodies.get("femur_r").getPositionInGround(state)
    shoulder = bodies.get("humerus_r").getPositionInGround(state)
    return {
        "points": points,
        "trunk_rad": float(
            np.arctan2(shoulder.get(0) - hip.get(0), shoulder.get(1) - hip.get(1))
        ),
        "bar_y_seated": float(bar.get(1) - points[:, 1].min())
        - POSE_FLOOR_PENETRATION_M,
        "bar_over_midfoot": float(bar.get(0) - right[:, 0].mean()),
        "arm_lean": float(
            bodies.get("hand_r").getPositionInGround(state).get(0) - shoulder.get(0)
        ),
        "foot_pitch": float(right[0, 1] - right[1:, 1].mean()),
    }


def _solve_pose(
    model: Any,
    state: Any,
    feet: list[Any],
    pose: dict[str, float],
    free: tuple[str, ...],
    equations: Any,
) -> dict[str, float]:
    """Solve the ``free`` coordinates so that ``equations(features)`` vanish."""

    cset = model.getCoordinateSet()
    names = {k: f"{k}_r" if k in (*_LEG_POSE_KEYS, "arm_flex") else k for k in free}
    lower = [cset.get(names[k]).getRangeMin() for k in free]
    upper = [cset.get(names[k]).getRangeMax() for k in free]

    def residual(x: np.ndarray) -> np.ndarray:
        _apply_pose(model, state, {**pose, **dict(zip(free, x, strict=True))})
        return np.asarray(equations(_pose_features(model, state, feet)))

    x0 = np.clip([pose[k] for k in free], lower, upper)
    fit = least_squares(residual, x0, bounds=(lower, upper), xtol=1e-12)
    worst = float(np.abs(fit.fun).max())
    require(
        worst <= POSE_SOLVE_TOLERANCE_M,
        f"deadlift pose solve for {free} left a residual of {worst} m",
    )
    return {**pose, **dict(zip(free, (float(v) for v in fit.x), strict=True))}


def _seat_pose(
    model: Any,
    state: Any,
    feet: list[Any],
    pose: dict[str, float],
    foot_x_m: float,
) -> dict[str, Any]:
    """Translate a pose onto the floor with its right foot centred on ``foot_x_m``."""
    pose = {**pose, "pelvis_tx": 0.0, "pelvis_ty": 0.0}
    _apply_pose(model, state, pose)
    points = _ground_points(model, state, feet)
    pose["pelvis_tx"] = foot_x_m - float(points[:3, 0].mean())
    pose["pelvis_ty"] = -float(points[:, 1].min()) - POSE_FLOOR_PENETRATION_M
    _apply_pose(model, state, pose)
    points = _ground_points(model, state, feet)
    bar = model.getBodySet().get("barbell").getPositionInGround(state)
    return {
        **pose,
        "bar_height_m": float(bar.get(1)),
        "bar_x_m": float(bar.get(0)),
        "com_x_m": float(model.calcMassCenterPosition(state).get(0)),
        "sphere_x_m": points[:, 0].tolist(),
        "sphere_y_m": points[:3, 1].tolist(),
    }


def compute_deadlift_poses(
    model: Any, targets: dict[str, float], height_m: float = 1.78
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Statically consistent floor-setup and lockout poses (OSL-7).

    ``targets`` comes from ``bundle_pose_targets``.  Both poses stand
    flat-footed on the same planted feet with straight arms hanging
    vertically, and incline the trunk (hip to shoulder) as the bundle's
    optimizer does.  The lockout extends hips and knees and solves pelvis
    tilt, ankle and shoulder flexion; the setup also solves hip and knee to
    put the bar at the bundle's starting height over the midfoot.
    Postcondition: every equation residual is at most ``POSE_SOLVE_TOLERANCE_M``.
    """
    for key in ("setup_trunk_rad", "lockout_trunk_rad", "setup_bar_height_m"):
        require(key in targets, f"targets lacks {key!r}")
    feet = _foot_spheres(height_m / 1.70)
    state = model.initSystem()
    cset = model.getCoordinateSet()
    base = {
        "hip_flexion": 0.0,
        "knee_angle": 0.0,
        "ankle_angle": float(cset.get("ankle_angle_r").getDefaultValue()),
        "pelvis_tilt": float(cset.get("pelvis_tilt").getDefaultValue()),
        "lumbar_extension": 0.0,
        "arm_flex": float(cset.get("arm_flex_r").getDefaultValue()),
        "pelvis_tx": 0.0,
        "pelvis_ty": 0.0,
    }
    lockout = _solve_pose(
        model,
        state,
        feet,
        base,
        ("pelvis_tilt", "ankle_angle", "arm_flex"),
        lambda f: (
            f["foot_pitch"],
            f["trunk_rad"] - targets["lockout_trunk_rad"],
            f["arm_lean"],
        ),
    )
    setup = _solve_pose(
        model,
        state,
        feet,
        {**lockout, "hip_flexion": 1.2, "knee_angle": 1.2, "arm_flex": 0.8},
        ("hip_flexion", "knee_angle", "ankle_angle", "pelvis_tilt", "arm_flex"),
        lambda f: (
            f["bar_y_seated"] - targets["setup_bar_height_m"],
            f["bar_over_midfoot"],
            f["trunk_rad"] - targets["setup_trunk_rad"],
            f["arm_lean"],
            f["foot_pitch"],
        ),
    )
    return (
        _seat_pose(model, state, feet, setup, 0.0),
        _seat_pose(model, state, feet, lockout, 0.0),
    )


#: Half-width of the initial and final value bounds around a solved pose.
POSE_BOUND_RAD = 0.05
POSE_BOUND_M = 0.01
#: Path margin of the pelvis translations around the two poses (m).
PATH_MARGIN_M = 0.1
_POSE_COORDINATES = {
    "hip_flexion": ("hip_flexion_r", "hip_flexion_l"),
    "knee_angle": ("knee_angle_r", "knee_angle_l"),
    "ankle_angle": ("ankle_angle_r", "ankle_angle_l"),
    "arm_flex": ("arm_flex_r", "arm_flex_l"),
    "pelvis_tilt": ("pelvis_tilt",),
    "lumbar_extension": ("lumbar_extension",),
    "pelvis_tx": ("pelvis_tx",),
    "pelvis_ty": ("pelvis_ty",),
}


def planted_feet_reference(
    model: Any, setup_pose: dict[str, float]
) -> dict[str, list[float]]:
    """Ground-frame origin of each calcaneus in the solved setup pose."""
    state = model.initSystem()
    _apply_pose(model, state, setup_pose)
    bodies = model.getBodySet()
    return {
        path: [
            float(v)
            for v in np.asarray(
                bodies.get(path.rsplit("/", 1)[1]).getPositionInGround(state).to_numpy()
            )
        ]
        for path in PLANTED_FEET_FRAMES
    }


def _add_planted_feet_goal(
    problem: Any, reference: dict[str, list[float]], t_end: float, weight: float
) -> list[float]:
    """Track both calcanei at fixed ground positions over ``[0, t_end]``.

    A deadlift keeps the feet planted; without this goal the optimizer lifts
    them and carries the body on the pelvis residual instead of the ground.
    Returns the reference sample times (postcondition: they span ``t_end``).
    """
    require(t_end > 0.0, f"t_end must be positive: {t_end}")
    osim = import_opensim()
    labels = osim.StdVectorString()
    for path in PLANTED_FEET_FRAMES:
        labels.append(path)
    table = osim.TimeSeriesTableVec3()
    table.setColumnLabels(labels)
    times = np.linspace(0.0, t_end + PLANTED_FEET_TIME_MARGIN_S, PLANTED_FEET_SAMPLES)
    for t in times:
        row = osim.RowVectorVec3(len(PLANTED_FEET_FRAMES))
        for i, path in enumerate(PLANTED_FEET_FRAMES):
            row[i] = osim.Vec3(*reference[path])
        table.appendRow(float(t), row)
    goal = osim.MocoTranslationTrackingGoal("planted_feet", weight)
    goal.setTranslationReference(table)
    goal.setFramePaths(labels)
    problem.addGoal(goal)
    return [float(t) for t in times]


def _clip_bound(value: float, width: float, path: list[float]) -> list[float]:
    """``value`` plus or minus ``width``, clipped to the path bounds."""
    return [max(path[0], value - width), min(path[1], value + width)]


def _configure_phase_bounds(
    phase: Any,
    model: Any,
    setup_pose: dict[str, Any],
    lockout_pose: dict[str, Any],
) -> None:
    """Bound the phase start and end on the solved setup and lockout poses.

    Rotations may move over their model range along the path; the pelvis
    translations over the two poses' envelope widened by ``PATH_MARGIN_M``.
    Coordinates in ``ZERO_CONSTRAINED_COORDS`` are held at zero throughout.
    """
    cset = model.getCoordinateSet()
    for key, coords in _POSE_COORDINATES.items():
        start, end = float(setup_pose[key]), float(lockout_pose[key])
        width = POSE_BOUND_M if key.startswith("pelvis_t") else POSE_BOUND_RAD
        for name in coords:
            coord = cset.get(name)
            if key.startswith("pelvis_t"):
                path = [
                    min(start, end) - PATH_MARGIN_M,
                    max(start, end) + PATH_MARGIN_M,
                ]
            else:
                path = [coord.getRangeMin(), coord.getRangeMax()]
            prefix = f"/jointset/{coord.getJoint().getName()}/{name}"
            phase.setStateInfo(
                f"{prefix}/value",
                path,
                _clip_bound(start, width, path),
                _clip_bound(end, width, path),
            )
            phase.setStateInfo(
                f"{prefix}/speed", [-10.0, 10.0], [0.0, 0.0], [-0.5, 0.5]
            )

    for z in ZERO_CONSTRAINED_COORDS:
        joint_name = cset.get(z).getJoint().getName()
        phase.setStateInfo(f"/jointset/{joint_name}/{z}/value", [0.0, 0.0], 0.0, 0.0)
        phase.setStateInfo(f"/jointset/{joint_name}/{z}/speed", [0.0, 0.0], 0.0, 0.0)


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

    set_actuator_optimal_force(
        model, SHOULDER_ACTUATORS, config.torque_optimal_force_nm
    )
    _add_pelvis_residuals(model, config.residual_optimal_force_n())
    pose_targets = bundle_pose_targets()
    setup_pose, lockout_pose = compute_deadlift_poses(
        model, pose_targets, height_m=config.height_m
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
    for coord in PELVIS_RESIDUAL_COORDINATES:
        effort.setWeightForControl(
            f"/forceset/residual_{coord}", config.residual_effort_weight
        )
    problem.addGoal(effort)
    feet_reference = planted_feet_reference(model, setup_pose)
    feet_times = _add_planted_feet_goal(
        problem, feet_reference, config.duration_bounds[1], config.planted_feet_weight
    )
    _configure_phase_bounds(phase, model, setup_pose, lockout_pose)

    solver = osim.MocoCasADiSolver.safeDownCast(study.initCasADiSolver())
    solver.set_num_mesh_intervals(config.mesh_intervals)
    solver.set_transcription_scheme("hermite-simpson")
    solver.set_multibody_dynamics_mode(config.multibody_dynamics_mode)
    solver.set_optim_finite_difference_scheme(config.finite_difference_scheme)
    solver.set_optim_max_iterations(config.max_iterations)
    solver.set_optim_convergence_tolerance(config.convergence_tolerance)
    solver.set_optim_constraint_tolerance(config.constraint_tolerance)

    meta = create_predictive_deadlift_spec(config)
    meta["model_name"] = model.getName()
    meta["pose_targets"] = pose_targets
    meta["setup_pose"] = setup_pose
    meta["lockout_pose"] = lockout_pose
    meta["planted_feet_reference"] = feet_reference
    meta["planted_feet_times"] = feet_times
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
        solution, tot_weight, config.residual_optimal_force_n()
    )
    moment_peak, moment_rms = residual_moment_metrics(
        solution, config.residual_optimal_force_n()
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
        "residual_moment_peak_nm": moment_peak,
        "residual_moment_rms_nm": moment_rms,
    }

    if converged:
        try:
            record.update(_extract_bar_kinematics(study, solution))
        except (RuntimeError, ValueError) as exc:
            logger.warning("FK replay failed on converged solution: %s", exc)
    gate = residual_moment_gate(moment_peak, record.get("com_height_m"), tot_weight)
    record.update(gate)
    if gate["residual_moment_within_limit"] is False:
        record["residual_assisted"] = True

    return record


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
        "residual_optimal_force_n": config.residual_optimal_force_n(),
        "residual_effort_weight": config.residual_effort_weight,
        "planted_feet_weight": config.planted_feet_weight,
        "residual_moment_peak_nm": solve_summary.get("residual_moment_peak_nm"),
        "residual_moment_rms_nm": solve_summary.get("residual_moment_rms_nm"),
        "residual_moment_limit_nm": solve_summary.get("residual_moment_limit_nm"),
        "residual_moment_within_limit": solve_summary.get(
            "residual_moment_within_limit"
        ),
        "converged": solve_summary.get("converged", False),
        "solver_status": solve_summary.get("solver_status", "unknown"),
        "iterations": solve_summary.get("iterations"),
        "objective": solve_summary.get("objective", float("nan")),
        "constraint_violation": solve_summary.get("constraint_violation"),
        "last_iterate_inf_pr": solve_summary.get("last_iterate_inf_pr"),
        "wall_clock_s": solve_summary.get("wall_clock_s"),
        "pose_targets": solve_summary.get("pose_targets"),
        "setup_pose": solve_summary.get("setup_pose"),
        "lockout_pose": solve_summary.get("lockout_pose"),
        "comparison_table": comp_table,
        "reproduction_command": "python scripts/measure_opensim_predictive_deadlift.py",
    }
